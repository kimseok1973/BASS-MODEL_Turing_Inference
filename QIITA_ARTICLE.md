# Julia で Bass 普及モデルをベイズ推定する — ModelingToolkit.jl × Turing.jl

## モチベーション

新製品の普及予測をするとき、Bass モデルを使うことがある。教科書的には最小二乗法でパラメータを求めることが多いが、「推定値の不確実性はどのくらいか？」「将来予測の信用区間は？」という問いには答えられない。

ベイズ推定なら事後分布としてパラメータの不確実性をそのまま扱えるし、事後サンプルを使って将来予測の信用区間も自然に出せる。

というわけで、Julia の `ModelingToolkit.jl` で Bass モデルの ODE を定義し、`Turing.jl` の NUTS サンプリングでスマートフォン普及率データにフィットしてみた。メモ代わりに記しておく。

---

## Bass モデルとは

Frank Bass (1969) が提唱した新製品普及の微分方程式モデル。累積採用者数 $N(t)$ の変化率を以下で表す。

$$\frac{dN}{dt} = \left( p + q \cdot \frac{N}{M} \right)(M - N)$$

| 記号 | 意味 |
|------|------|
| $p$ | イノベーション係数（広告など外部影響） |
| $q$ | イミテーション係数（口コミなど内部影響） |
| $M$ | 市場ポテンシャル（最終的な採用者総数） |

- $p$ が大きい → 広告主導で初期から普及が進む
- $q$ が大きい → 口コミで爆発的に広がるが初期は遅い

右辺は「イノベーター項」と「イミテーター項」の和になっており、累積普及率が上がるほどイミテーター効果が強まる構造になっている。

---

## ModelingToolkit.jl で ODE を定義する

Julia の `ModelingToolkit.jl` を使うと、ODE をシンボリックに記述してコンパイルできる。数値解析に最適化された形に自動変換してくれるので、手でコードを書く必要がない。

```julia
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using DifferentialEquations
using Plots

@parameters p q M
@variables N(t)   # 累積採用者数
@variables n(t)   # 新規採用者数（観測量）

eqs = [
    D(N) ~ (p + q * N / M) * (M - N),
    n    ~ (p + q * N / M) * (M - N)
]

@named bass = ODESystem(eqs, t)
sys = mtkcompile(bass)
```

パラメータと初期値を `Dict` でまとめて渡すのが現行 API の推奨形式。

```julia
tspan = (0.0, 20.0)
prob = ODEProblem(sys,
    Dict(N => 0.0, p => 0.03, q => 0.38, M => 1000.0),
    tspan)
sol = solve(prob, Tsit5(); saveat = 0.1)
```

S字カーブとベル型の普及速度が得られる（グラフは省略）。

---

## データ：スマートフォン普及率（2010〜2024年）

総務省「情報通信白書」から、日本のスマートフォン世帯普及率を使う。

```julia
years_obs = 2010:2024
obs_rate  = [4.4, 21.1, 22.9, 36.8, 46.7, 51.1, 59.8, 71.7,
             74.3, 85.0, 88.9, 92.8, 94.0, 96.3, 97.0]
t_obs     = Float64.(0:14)    # 2010 → t=0, 2024 → t=14
```

2010年時点で既に 4.4% の普及があり、2024年には 97.0% まで到達している。典型的なS字カーブが見える。

---

## Turing.jl でベイズ推定

`Turing.jl` の `@model` マクロ内で ODE を毎回解く。パラメータをサンプリングするたびに `ODEProblem` を生成して `solve` を呼ぶ構造になる。

### ODE の直接定義

`ModelingToolkit.jl` でコンパイル済みの `sys` を使う方法もあるが、Turing と組み合わせるときは素直に `DifferentialEquations.jl` の標準形で書いたほうがシンプル。

```julia
function bass_ode!(du, u, params, t)
    p_v, q_v, M_v = params
    N = u[1]
    du[1] = (p_v + q_v * N / M_v) * (M_v - N)
end
```

### ベイズモデルの定義

```julia
using Turing

@model function bass_bayes(t_data, y_data)
    # 事前分布
    p_v ~ truncated(Normal(0.02, 0.02); lower = 1e-4, upper = 0.5)
    q_v ~ truncated(Normal(0.30, 0.20); lower = 1e-4, upper = 2.0)
    M_v ~ truncated(Normal(100.0, 5.0); lower = maximum(y_data) + 0.1, upper = 115.0)
    σ   ~ Exponential(3.0)

    N0 = y_data[1]   # 初期値：2010年の観測値（4.4%）を固定

    prob_ode = ODEProblem(bass_ode!, [N0], (t_data[1], t_data[end]), [p_v, q_v, M_v])
    sol = solve(prob_ode, Tsit5(); saveat = t_data, abstol = 1e-6, reltol = 1e-6)

    if !SciMLBase.successful_retcode(sol)
        Turing.@addlogprob! -Inf
        return
    end

    N_pred = sol[1, :]

    # 尤度（初期値は IC として使用済みなので 2 点目から）
    for i in 2:length(y_data)
        y_data[i] ~ Normal(N_pred[i], σ)
    end
end
```

設計のポイントをまとめると：

- `M` の下限を `maximum(y_data) + 0.1` にする → 観測最大値より必ず大きくなるよう制約
- ODE が解けなかった場合は `@addlogprob! -Inf` でそのサンプルを棄却
- 初期値 `N0 = 4.4` を固定して `i=2` から尤度を評価することで、初期条件の情報二重使用を避ける

### サンプリング

```julia
model_bass = bass_bayes(t_obs, obs_rate)
chain_bass = sample(model_bass, NUTS(0.65), 1000)
describe(chain_bass)
```

`NUTS(0.65)` の `0.65` は目標受容率（target acceptance rate）。デフォルトの `0.8` より少し低めにして探索の幅を確保している。1000 サンプルなら数分で終わる。

---

## 推定結果

事後平均パラメータの例（実行ごとに多少変わる）：

| パラメータ | 事後平均 | 解釈 |
|-----------|---------|------|
| `p` | 約 0.008 | イノベーション係数：広告効果は小さい |
| `q` | 約 0.55  | イミテーション係数：口コミ効果が支配的 |
| `M` | 約 99 %  | 市場ポテンシャル：ほぼ全世帯に普及 |

日本のスマートフォン普及は $q \gg p$ の典型的な「口コミ主導型」であることが定量的に確認できた。

事後予測プロットでは 90% 信用区間バンドが観測値をほぼカバーしており、Bass モデルのフィットはかなり良い。

---

## 事後予測の実装

推定済みチェーンから全サンプル分の ODE を再解し、分位点から信用区間を作る。

```julia
using Statistics

p_samp = vec(chain_bass[:p_v])
q_samp = vec(chain_bass[:q_v])
M_samp = vec(chain_bass[:M_v])

t_fine     = collect(0.0:0.1:20.0)
N0_val     = obs_rate[1]
pred_mat   = fill(NaN, length(p_samp), length(t_fine))

for i in eachindex(p_samp)
    sol_i = solve(
        ODEProblem(bass_ode!, [N0_val], (t_fine[1], t_fine[end]),
                   [p_samp[i], q_samp[i], M_samp[i]]),
        Tsit5(); saveat = t_fine)
    if SciMLBase.successful_retcode(sol_i)
        pred_mat[i, :] = sol_i[1, :]
    end
end

med_pred = [median(filter(!isnan, pred_mat[:, j])) for j in 1:length(t_fine)]
lo_pred  = [quantile(filter(!isnan, pred_mat[:, j]), 0.05) for j in 1:length(t_fine)]
hi_pred  = [quantile(filter(!isnan, pred_mat[:, j]), 0.95) for j in 1:length(t_fine)]
```

あとは `ribbon=` オプションで帯を描くだけ。

```julia
scatter(collect(years_obs), obs_rate; label = "観測値", ms = 7)
plot!(2010.0 .+ t_fine, med_pred;
    ribbon   = (med_pred .- lo_pred, hi_pred .- med_pred),
    fillalpha = 0.25, lw = 2, color = :crimson,
    label = "事後予測中央値（90%信用区間）",
    xlabel = "年", ylabel = "普及率（%）")
```

---

## まとめ

- `ModelingToolkit.jl` で Bass ODE をシンボリックに定義 → 感度分析や拡張（時変 p、複数セグメント合成）が容易
- `Turing.jl` の `@model` 内で `ODEProblem` を毎回解くことで ODE のベイズ推定が素直に書ける
- 日本のスマートフォン普及は $q \approx 0.55,\; p \approx 0.008$ の口コミ主導型と推定された

ノートブック全体（感度分析・広告時変モデル・2セグメント合成モデル含む）は GitHub に置いてある。

https://github.com/kimseok1973/BASS-MODEL_Turing_Inference

以上。
