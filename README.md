# Bass普及モデル — ModelingToolkit.jl × Turing.jl

Bass (1969) の新製品普及モデルを Julia で実装し、スマートフォン普及率データへのベイズパラメータ推定を行うノートブックです。

## モデル概要

$$\frac{dN}{dt} = \left( p + q \cdot \frac{N}{M} \right)(M - N)$$

| 記号 | 意味 |
|------|------|
| $N(t)$ | 累積採用者数（または普及率） |
| $M$ | 市場ポテンシャル |
| $p$ | イノベーション係数（広告・外部影響） |
| $q$ | イミテーション係数（口コミ・内部影響） |

## ノートブック構成

| セクション | 内容 |
|-----------|------|
| 1. モデル定義 | `ModelingToolkit.jl` で ODE をシンボリックに記述 |
| 2. 基本シミュレーション | S字カーブ・ベル型普及速度の描画 |
| 3. 感度分析 — q | イミテーション係数の影響 |
| 4. 感度分析 — p | イノベーション係数の影響 |
| 5. 広告時変モデル | `@register_symbolic` で p(t) を外部関数として組み込み |
| 6. 2セグメント合成 | アーリー・レイト採用者を独立コンポーネントで合成 |
| 7. p-q 位相空間マップ | 普及ピーク到達時刻のヒートマップ |
| 8. **ベイズ推定** | `Turing.jl` NUTS サンプリングでスマートフォン普及率データに当てはめ |

## ベイズ推定の設計

- **データ**：スマートフォン世帯普及率 2010〜2024年（総務省 情報通信白書）
- **サンプラー**：NUTS（No-U-Turn Sampler）
- **事前分布**

| パラメータ | 事前分布 |
|-----------|---------|
| `p` | $\text{Truncated-Normal}(0.02,\, 0.02)$ |
| `q` | $\text{Truncated-Normal}(0.30,\, 0.20)$ |
| `M` | $\text{Truncated-Normal}(100,\, 5)$ |
| `σ` | $\text{Exponential}(3)$ |

- 初期値（2010年 = 4.4%）を固定し、2011〜2024年の14点で尤度を評価
- 事後サンプル全件で ODE を再解し、90%信用区間バンドを描画

## 使用パッケージ

```julia
using ModelingToolkit, DifferentialEquations, Turing
using Plots, StatsPlots, Statistics
```

## 参考文献

- Bass, F. M. (1969). A new product growth for model consumer durables. *Management Science*, 15(5), 215–227.
- 総務省「令和6年版 情報通信白書」スマートフォン普及率データ
- [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)
- [Turing.jl](https://github.com/TuringLang/Turing.jl)
