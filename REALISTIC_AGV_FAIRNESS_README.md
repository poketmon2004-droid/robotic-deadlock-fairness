# Realistic AGV Simulation & Fairness Experiments

## 연구 목적
기존 BO·PSO 기반 AGV 진입 우선순위 최적화 연구를
물리적 이동시간과 대기 공정성 측면에서 확장한다.

## 주요 개선 사항
1. 거리·속도·회전시간 기반 AGV 이동 모델
2. AGV별 주행속도 차이 반영
3. 대기열 중복 등록 방지
4. FairBO·FairPSO 장시간 대기 보호 규칙
5. 기존·신규 72개 시나리오 성능 검증

## 시뮬레이션 설정
- Grid cell: 1.5 m
- Base speed: 1.0 m/s
- Angular speed: 1.0 rad/s
- Simulation timestep: 0.1 s
- AGV counts: 20, 24, 28
- Scenario types: S5, S6, S7, S8

위 물리적 수치는 시뮬레이션 가정값이다.

## 핵심 코드
- rware/realistic_agv_config.py
- rware/realistic_cross_simulation.py
- analyze_step93.py
- analyze_step96_queue_fix.py
- analyze_step96_fair_bo.py
- analyze_step99_fairness.py
- analyze_step100_holdout.py
- analyze_step103_original.py
- analyze_step103_holdout.py
- analyze_step104.py

## 최종 결과
총 72개 시나리오에서:

- FairBO: BO 대비 최대 대기시간 평균 48.3% 감소,
  전체 완료시간 0.76% 증가
- FairPSO: PSO 대비 최대 대기시간 평균 48.4% 감소,
  전체 완료시간 0.88% 증가

## 검증 범위
- 단일 교차로 기반 시뮬레이션
- 기존 36개 + 신규 36개 시나리오
- 대응표본 부트스트랩 5,000회
- 실제 AGV 운행 검증은 수행하지 않음

## 주의
기존 종설 결과와 새로운 물리 모델 결과는
시뮬레이션 조건이 다르므로 직접 수치 비교하지 않는다.
