import uuid
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from deap import creator, base, tools, algorithms

# GA-ML 알고리즘 준비
# Log Loss를 기반으로 한 평가 함수 (적합도) 정의
def evaluate(individual, X_train, y_train, model):
    selected_features = [index for index, value in enumerate(individual) if value == 1]

    # 선택된 특성이 없을 경우 큰 오류 값을 반환
    if len(selected_features) == 0:
        return 1e6,

    # 선택된 특성으로 모델 학습
    model.fit(X_train[:, selected_features], y_train)
    y_pred_prob = model.predict_proba(X_train[:, selected_features])[:, 1]

    # Log Loss 계산
    loss = log_loss(y_train, y_pred_prob)

    return loss,

# 임의의 개체 생성 (특성 선택을 나타내는 유전자)
def create_individual(n_features):
    return np.random.randint(0, 2, n_features).tolist()

# 초기 인구 생성
def create_population(n_individuals, n_features):
    return [create_individual(n_features) for _ in range(n_individuals)]

# 유전자 알고리즘 설정
def ga_feature_selection(X, y, stock_ticker, model, population_size=50, generations=30, crossover_prob=0.5, mutation_prob=0.375):
    n_features = X.shape[1]
    unique_id = str(uuid.uuid4())  # 고유 ID 생성
    creator.create(f"MyFitnessMin_{stock_ticker}_{unique_id}", base.Fitness, weights=(-1.0,))
    creator.create(f"MyIndividual_{stock_ticker}_{unique_id}", list, fitness=creator.__getattribute__(f"MyFitnessMin_{stock_ticker}_{unique_id}"))

    # 툴박스 설정
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", create_individual, n_features)
    toolbox.register("individual", tools.initIterate, creator.__getattribute__(f"MyIndividual_{stock_ticker}_{unique_id}"), toolbox.attr_bool)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

     # 평가 함수 등록
    toolbox.register("evaluate", evaluate, X_train=X, y_train=y, model=model)

    population = toolbox.population(n=population_size)

    best_loss = float('inf')
    best_individual = None

    hof = tools.HallOfFame(1)

    # 유전자 알고리즘 실행
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
        fits = list(map(toolbox.evaluate, offspring))

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        # 현재 인구와 자손을 결합
        combined_population = population + offspring

        # 다음 세대 인구 선택
        population = toolbox.select(combined_population, k=population_size)

        # 최적의 개체를 Hall of Fame에 업데이트
        hof.update(population)

        # 현재 세대의 최적 개체 및 Log Loss 찾기
        current_best_individual = hof[0]
        current_best_loss = current_best_individual.fitness.values[0]

        if current_best_loss < best_loss:
            best_loss = current_best_loss
            best_individual = current_best_individual

        # 선택된 특성 인덱스
        selected_features = [index for index, value in enumerate(best_individual) if value == 1]
        if (gen + 1) == generations:
            print(f"Stock: {stock_ticker} Generation {gen + 1}: Best Log Loss = {best_loss:.5f}, Selected Features = {selected_features}")

    return best_individual

# 전역 변수로 사용될 feature_importances 초기화
feature_importances_global = None
def show_result(X, y, model, stock_ticker):
    global feature_importances_global

    if feature_importances_global is None:
        feature_importances_global = np.zeros(X.shape[1])

    # 유전자 알고리즘 실행하여 특성 선택
    best_individual = ga_feature_selection(X, y, stock_ticker, model)

    # 최종 선택된 특성 인덱스 출력
    final_selected_features = [index for index, value in enumerate(best_individual) if value == 1]
    print(f"Final Selected Features({stock_ticker}): {final_selected_features}")

    # 로컬 특성 중요도 배열 초기화
    feature_importances = np.zeros(X.shape[1])

    # 선택된 특성으로 모델 학습
    model.fit(X[:, final_selected_features], y)

    # 모델에서 선택된 특성의 중요도 점수 가져오기
    selected_feature_importances = model.feature_importances_

    # 로컬 배열에 선택된 특성 중요도 할당
    for idx, feature_index in enumerate(final_selected_features):
        feature_importances[feature_index] = selected_feature_importances[idx]

    # 상위 5개 특성의 인덱스를 중요도에 따라 내림차순으로 정렬
    values = np.array(feature_importances)
    top_5_indices = np.argsort(values)[-5:][::-1]
    top_5_values = values[top_5_indices]

    print(f"Selected Top5 Features by Importance Score({stock_ticker})")
    # 상위 5개 인덱스와 해당 값을 출력
    for idx, val in zip(top_5_indices, top_5_values):
        print(f"Feature: {idx}, Importance Score: {val}")

    # 전역 변수에 특성 중요도 누적
    feature_importances_global += feature_importances

    # 선택된 특성으로 모델 학습
    model.fit(X[:, final_selected_features], y)
    y_pred_prob = model.predict_proba(X[:, final_selected_features])[:, 1]

    # 전체 학습 데이터에서 최종 Log Loss 계산
    final_loss = log_loss(y, y_pred_prob)

    return final_loss, best_individual

def calculate_final_importance(n_stocks):
    # 평균 특성 중요도 계산
    average_feature_importance = feature_importances_global / n_stocks

    # 데이터프레임으로 변환하여 순위 매기기
    feature_importance_df = pd.DataFrame({
        'Feature': range(len(average_feature_importance)),
        'Importance': average_feature_importance
    })

    # 평균 중요도를 기준으로 특성 정렬
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("Feature ranking based on average importance:")
    print(feature_importance_df)

    return feature_importance_df
