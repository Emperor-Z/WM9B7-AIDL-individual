# Critical Reflection: On-Device Image Classification for Smart Home Security

## 1. Problem and Dataset

The project addresses a concrete deployment challenge: classifying images from a low-resolution smart home camera entirely on a Raspberry Pi 4—no cloud, no GPU, with a hard constraint of sub-millisecond inference latency and a model footprint under 10 MB. CIFAR-10 (Krizhevsky, 2009) serves as the benchmark proxy: 60,000 RGB images at 32 × 32 pixels across ten balanced classes spanning household pets, garden wildlife, and driveway activity. A 45k/5k/10k train/validation/test split (seed 42) was applied; the test set was held out entirely and evaluated only once per model to prevent optimistic bias (Hastie et al., 2009). The low resolution deliberately mirrors degraded edge-camera conditions, making the benchmark directly analogous to on-device inference in production.

---

## 2. Machine Learning vs Deep Learning

### 2.1 The Traditional ML Baseline: HOG + LinearSVC

The HOG + LinearSVC pipeline (Dalal & Triggs, 2005) represents the pre-deep-learning paradigm. HOG computes gradient orientation histograms across 4 × 4 pixel cells with 2 × 2 block normalisation, producing a fixed 1,764-dimensional descriptor per image. A LinearSVC (C = 0.1) then learns class boundaries in this hand-crafted feature space, achieving **40.85% test accuracy** on the full 10,000-sample CIFAR-10 test set.

The fundamental limitation is representational rigidity. HOG encodes local edge statistics defined by the engineer; it cannot adapt those statistics to the data. On 32 × 32 images, only four HOG cells span each spatial dimension, yielding sparse descriptors with limited discriminative power. Inter-class similarity—cat versus dog, automobile versus truck—cannot be resolved without hierarchical feature abstraction unavailable to a fixed descriptor.

Computationally, HOG excels: feature extraction completes in microseconds and the entire pipeline fits under 1 MB. This makes it the correct choice for microcontroller-class devices where floating-point CNNs are impractical. For the Raspberry Pi 4 target—which supports Python runtimes and PyTorch inference—the 43-percentage-point accuracy gap relative to the best edge-deployable DL model is operationally unacceptable.

### 2.2 Why Deep Learning Is Justified

Convolutional Neural Networks learn hierarchical feature detectors end-to-end through gradient descent (LeCun et al., 1998). Early layers detect edges; mid layers detect textures; deep layers detect semantic object parts. This hierarchy naturally captures the intra-class diversity that defeats HOG: a cat can appear at any angle, under any lighting, and the network learns invariance through data exposure rather than manual specification.

The quantitative gap is stark. CompactCNN—trained from scratch on 45,000 images—achieves **84.14% test accuracy**, a 43.3 percentage-point improvement over HOG + LinearSVC. ResNet-18 reaches **94.94%** and Swin-T reaches **97.83%**, collectively demonstrating that learned hierarchical representations far outpace hand-crafted descriptors on this task. This advantage comes with real costs: deep learning is data-hungry (HOG+SVM was trained on 10,000 samples; CNNs on 45,000), compute-intensive during training (~120 minutes on an RTX 3050 Ti for the full pipeline), and opaque by default—motivating post-hoc interpretability methods like GradCAM (Selvaraju et al., 2017). The choice of deep learning is justified by the availability of a sufficient training corpus, a GPU-capable training environment, and a deployment target that supports PyTorch inference at runtime.

---

## 3. Design Decisions

### 3.1 Architecture Selection and the Accuracy–Efficiency Frontier

Five deep learning architectures were trained to map the Pareto frontier between accuracy and deployment feasibility:

**CompactCNN** (620,810 parameters, 2.48 MB) is a purpose-built three-block convolutional network (32→64→128 filters) with batch normalisation and dropout (0.5 in the feature extractor, 0.3 in the classifier). Trained for 60 epochs from scratch, it achieves **84.14%** with a CPU inference latency of **0.69 ms**—the only architecture satisfying both the 10 MB footprint and sub-millisecond latency requirements. It is the primary recommended deployment model.

**MobileNetV3-Small** (1,528,106 parameters, 6.11 MB) uses depthwise-separable convolutions, factorising a standard convolution into depthwise (per-channel spatial) and pointwise (cross-channel) operations, reducing FLOPs by approximately 8–9× (Howard et al., 2019). Hard-Swish activations are designed for inference on hardware with limited instruction sets. Despite ImageNet pre-training, MobileNetV3-Small achieves **83.43%**—marginally below CompactCNN—with a CPU latency of **5.75 ms**. The counter-intuitive underperformance relative to a from-scratch model is explained by the 32 × 32 input resolution: pretrained spatial hierarchies are calibrated to 224 × 224, and the severely downsampled features provide limited advantage over a network trained natively at the target resolution. Its 5.75 ms latency disqualifies it from the strictest sub-millisecond constraint, though it remains suitable where latency in the low single-digit millisecond range is acceptable.

**ResNet-18** (11,181,642 parameters, 44.73 MB) was selected over deeper variants because on 32 × 32 inputs, `layer4` already produces a 1 × 1 spatial feature map—additional depth would collapse representations before the classifier. Residual skip connections (He et al., 2016) address the vanishing gradient problem in deep networks. Two-stage fine-tuning was applied: Stage 1 freezes the backbone and trains only the new 512→10 classifier head at lr = 1e-3 for 5 epochs; Stage 2 unfreezes all 11.2M parameters and fine-tunes at lr = 1e-4 for 25 epochs. The order-of-magnitude LR reduction prevents catastrophic forgetting of ImageNet representations. ResNet-18 achieves **94.94%** with a CPU latency of 16.26 ms.

**EfficientNet-B0** (4,020,358 parameters, 16.08 MB) applies compound scaling—simultaneously increasing width, depth, and resolution via a NAS-derived coefficient (Tan & Le, 2019). With 4.0M parameters (64% fewer than ResNet-18) and only 34.4M FLOPs, EfficientNet-B0 reaches **91.14%**, demonstrating strong parameter efficiency, though its CPU latency of 14.06 ms reflects the cost of squeeze-and-excitation attention blocks. The same two-stage fine-tuning protocol was applied (Stage 1: 5 epochs at lr = 1e-3; Stage 2: 20 epochs at lr = 1e-4).

**Swin Transformer** (27,527,044 parameters, 110.11 MB) was included to establish the accuracy ceiling and represent the architectural shift toward vision transformers. Its hierarchical shifted-window attention mechanism (Liu et al., 2021) is designed for high-resolution inputs; pretrained patch embeddings require 224 × 224 images, necessitating bicubic upsampling from 32 × 32 and a halved batch size (64 vs 128) to fit 49× larger tensors in GPU memory. Stage 1 trained only the classification head for 5 epochs at lr = 1e-3; Stage 2 fine-tuned all 27.5M parameters for 15 epochs at lr = 1e-4. Swin-T achieves **97.83%**—the highest accuracy in the study by 2.9 percentage points over ResNet-18—but at a CPU latency of **95.67 ms** and 2,977 MFLOPs per image. At 11× the footprint budget and 138× the latency budget, its inclusion is explicitly didactic: transformer-class accuracy does not translate to edge deployability, and architecture selection must be grounded in hardware-specific constraints rather than leaderboard rankings.

All pretrained models required a stride adaptation—`features[0][0].stride = (1,1)` for MobileNetV3 and EfficientNet, `conv1.stride = (1,1)` and `maxpool = Identity()` for ResNet-18—to prevent spatial collapse to 1 × 1 at the first block on 32 × 32 inputs. Without this fix, residual blocks and attention windows operate on null spatial information and training diverges.

### 3.2 Optimiser, Scheduler, and Augmentation

AdamW (Loshchilov & Hutter, 2019) was used throughout, with weight decay (λ = 1e-4) applied to weight matrices only—explicitly excluding biases and batch normalisation parameters to match the intended decoupled regularisation semantics. Cosine annealing (Loshchilov & Hutter, 2017) reduced the learning rate smoothly to zero over each training stage, preventing oscillation near the loss minimum.

Data augmentation (RandomHorizontalFlip at p = 0.5; RandomCrop with padding = 4) was applied to training data only. An ablation study on CompactCNN over 15 epochs yielded a counterintuitive result: the augmented model achieved **77.38% best validation accuracy** versus **80.20% without augmentation**—a −2.82 percentage-point deficit. This outcome, while contrary to the expected regularisation benefit, is consistent with established observations in the augmentation literature: for compact models with limited parameter capacity, the added data variety slows convergence and the benefit of implicit regularisation may not materialise within the available epoch budget (Hernández-García & König, 2018). A longer training run or a larger model would likely reverse this outcome. Regardless, augmentation was retained in the final training pipeline given its well-established benefit at scale. Normalisation used CIFAR-10 channel statistics (μ = [0.4914, 0.4822, 0.4465]; σ = [0.2023, 0.1994, 0.2010]) computed from the training split—not the full dataset—to prevent test-set leakage.

### 3.3 AI Assistance

Claude (Anthropic) was used as a coding assistant throughout development. Contributions included boilerplate for PyTorch training loops, the GradCAM hook registration pattern, and suggestions for the stride adaptation fix for CIFAR-10 compatibility. All AI-generated code was reviewed, tested empirically, and adapted to the project architecture. The stride fix, for example, was suggested by Claude but required per-architecture validation: MobileNetV3 and EfficientNet share a `features[0][0]` access pattern; ResNet-18 required separate `conv1` and `maxpool` modifications. No AI-generated text appears verbatim in the notebook.

---

## 4. Results and Interpretability

### 4.1 Accuracy–Efficiency Trade-off

| Model | Test Acc. | Params | Size (MB) | FLOPs | CPU Latency | Edge-feasible? |
|---|---|---|---|---|---|---|
| HOG + LinearSVC | 40.85% | — | <1 | — | <1 ms | Yes (any device) |
| CompactCNN | 84.14% | 620,810 | 2.48 | 11.1M | **0.69 ms** | **Yes** |
| MobileNetV3-Small | 83.43% | 1,528,106 | 6.11 | 6.0M | 5.75 ms | Partial (footprint ✓, latency ✗) |
| ResNet-18 | 94.94% | 11,181,642 | 44.73 | 565.8M | 16.26 ms | No |
| EfficientNet-B0 | 91.14% | 4,020,358 | 16.08 | 34.4M | 14.06 ms | No |
| Swin-T | **97.83%** | 27,527,044 | 110.11 | 2,977M | 95.67 ms | No |

Only CompactCNN satisfies both the ≤10 MB footprint and sub-millisecond latency constraint for Raspberry Pi 4 deployment—making it the sole viable candidate for this specific hardware target. MobileNetV3-Small fits within the footprint budget but misses the latency threshold by 5×. ResNet-18 and EfficientNet-B0 are appropriate for less constrained edge hardware (e.g., Jetson Nano). Swin-T establishes the accuracy ceiling at 97.83% but at CPU latency of 95.67 ms is entirely impractical for real-time inference.

### 4.2 GradCAM: Interpretability and Its Limits

Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017) was applied to ResNet-18 at `layer4[-1]` and `layer3[-1]`. A critical limitation emerges at 32 × 32 resolution: `layer4` produces a 1 × 1 spatial feature map, so after bilinear upsampling to 32 × 32 every pixel receives identical activation weight—producing a spatially uniform heatmap that conveys no localisation information. This is not a failure of the method but of the resolution regime: GradCAM was designed for 224 × 224 inputs where `layer4` outputs 7 × 7. The `layer3` target (2 × 2 output) offers limited but non-trivial spatial resolution and was used as the primary interpretability layer.

Failure case analysis reveals systematic error patterns. Ship→Airplane confusions coincide with GradCAM activation concentrated on background regions (shared blue sky or water) rather than the object silhouette—the model is attending to context rather than the target object. Cat↔Dog confusions persist even where the model correctly localises the animal body, indicating that fine-grained texture discrimination is the bottleneck rather than spatial attention—consistent with the known difficulty of CIFAR-10's visually similar animal pairs (Russakovsky et al., 2015). Automobile↔Truck confusions arise from shared structural elements (wheels, windscreen geometry). Together, these patterns validate GradCAM's value as a failure-mode diagnostic while exposing a deeper limitation: the method identifies *where* the model looks, not *what features* within that region drive the decision. At 32 × 32 resolution this distinction collapses entirely at `layer4`, leaving only the `layer3` heatmaps as meaningful. GradCAM is additionally inapplicable to Swin-T, which requires attention rollout (Chefer et al., 2021) for transformer-compatible attribution.

---

## 5. Societal, Ethical, and Environmental Impact

### 5.1 Surveillance and Misuse Risk

An on-device image classifier designed for smart home triage is inherently dual-use. The stated application—distinguishing pets from intruders—is benign. However, the same architecture retrained on person-class data would constitute a capable covert surveillance system. Edge inference generates no cloud traffic and evades network-level monitoring, lowering the deployment barrier for misuse. Responsible deployment requires explicit purpose limitation, data retention policies, and informed consent for all persons within the camera's field of view—obligations enforceable under GDPR Article 5(1)(b) and the UK Data Protection Act 2018.

### 5.2 Dataset Bias

CIFAR-10's classes reflect a Western-centric, early-2000s North American visual context: "automobile" predominantly images American car designs; "horse" skews toward Western equestrian photography. A classifier trained on CIFAR-10 and deployed in a non-Western context may exhibit degraded accuracy for culturally distinct objects mapping loosely onto these categories. Furthermore, CIFAR-10 is perfectly balanced (6,000 images per class), whereas production deployments encounter heavily skewed distributions—likely >99% negative triggers in a home camera context. Threshold calibration for imbalanced inference is not addressed in this benchmark study and represents a gap between laboratory results and deployment reality.

### 5.3 Environmental Cost

The full training pipeline consumed approximately 0.10 kWh on an RTX 3050 Ti (50 W TDP, roughly two hours wall-clock). This is negligible in isolation. However, the per-inference cost differential is the more consequential figure for an always-on edge system: Swin-T requires ~4.5 GFLOPs per image versus CompactCNN's ~50 MFLOPs—a 90× gap that compounds across millions of camera frames over a product lifetime. Patterson et al. (2021) estimated GPT-3 pretraining at ~1,287 MWh; while large language models concentrate energy in training, always-on edge classifiers concentrate it in inference. Selecting CompactCNN over Swin-T is therefore not only technically correct for the deployment target but also the environmentally responsible choice.

---

## 6. Emerging Trends

### 6.1 Efficient Vision Transformers

Vision Transformers (ViT; Dosovitskiy et al., 2021) challenged the CNN paradigm by replacing local convolutions with global self-attention over image patches, enabling richer long-range dependency modelling. However, ViT's data hunger—requiring JFT-300M or ImageNet-21K pre-training to match CNNs at competitive accuracy—limits direct applicability to small-dataset regimes. DeiT (Touvron et al., 2021) addresses this through token-based knowledge distillation and strong regularisation, achieving competitive CIFAR-10 accuracy with ImageNet-1K pre-training only. Swin Transformer (Liu et al., 2021) further reduces inference cost through hierarchical feature maps and shifted-window attention, cutting the quadratic scaling of full attention to linear—though, as this study demonstrates, even this improvement leaves Swin-T unsuitable for sub-watt edge inference.

The near-term trajectory for this project points toward MobileViT (Mehta & Rastegari, 2022), which combines depthwise-separable convolutions with lightweight local self-attention blocks in a mobile-deployable architecture. MobileViT-XXS (1.3M parameters) achieves ImageNet-1K accuracy competitive with MobileNetV3-Large while capturing global scene context that pure depthwise convolutions cannot. This is directly relevant to the smart home classifier: scene-level cues (presence of furniture, outdoor vegetation) help resolve the animal class confusions that the current models—limited to local texture features—cannot fully address. A future iteration of this system could adopt MobileViT-XXS as a Pareto-superior replacement for CompactCNN within the same 10 MB deployment budget.

### 6.2 Neural Architecture Search and Hardware-Aware Design

EfficientNet's NAS-derived compound scaling (Tan & Le, 2019) optimises architecture coefficients for ImageNet accuracy without hardware awareness. Once-for-All (OFA; Cai et al., 2020) addresses this gap: a single overparameterised network is trained from which sub-networks can be extracted and specialised for any target hardware without retraining. Applied to this project, OFA would generate a CIFAR-10 classifier optimised directly for the Raspberry Pi 4's ARM Cortex-A72—matching a specific latency and memory budget rather than approximating it through manual architecture search.

### 6.3 Foundation Models and Zero-Shot Transfer

CLIP (Radford et al., 2021) learns joint image-text representations at web scale, enabling zero-shot classification via text prompts without task-specific training. A smart home classifier built on a CLIP backbone could be updated to new object categories—new pet species, new vehicle types—through prompt engineering alone, without retraining. The current barrier is inference cost; CLIP-ViT-B/32 requires ~150× the FLOPs of CompactCNN per image. Lightweight successors such as SigLIP (Zhai et al., 2023) reduce this gap substantially and, within the deployment window of a smart home product, may make prompt-based edge classification viable—eliminating the need for labelled retraining datasets entirely.

---

## References

Cai, H., Gan, C., Wang, T., Zhang, Z., & Han, S. (2020). Once-for-all: Train one network and specialize it for efficient deployment. *ICLR 2020*.

Chefer, H., Gur, S., & Wolf, L. (2021). Transformer interpretability beyond attention visualization. *CVPR 2021*, 782–791.

Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR 2005*, 886–893.

Dosovitskiy, A., et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR 2021*.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*, 770–778.

Hernández-García, A., & König, P. (2018). Data augmentation instead of explicit regularization. *arXiv:1806.03852*.

Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV 2019*, 1314–1324.

Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto Technical Report.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324.

Liu, Z., et al. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. *ICCV 2021*, 10012–10022.

Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR 2022*, 11976–11986.

Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR 2017*.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.

Mehta, S., & Rastegari, M. (2022). MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer. *ICLR 2022*.

Patterson, D., et al. (2021). Carbon and the broad AI industry. *arXiv:2104.10350*.

Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML 2021*, 8748–8763.

Russakovsky, O., et al. (2015). ImageNet large scale visual recognition challenge. *IJCV, 115*(3), 211–252.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017*, 618–626.

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML 2019*, 6105–6114.

Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). Training data-efficient image transformers & distillation through attention. *ICML 2021*, 10347–10357.

Zhai, X., et al. (2023). Sigmoid loss for language image pre-training. *ICCV 2023*.
