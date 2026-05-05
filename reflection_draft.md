# Critical Reflection: On-Device Image Classification for Smart Home Security

---

## 1. Introduction and Problem

This project explores whether a compact deep learning model can classify images from a low-resolution camera in a resource-constrained edge AI setting. The intended application is a smart home or similar small environmental monitoring camera where images are processed locally on hardware such as a Raspberry Pi 4 rather than transmitted to a cloud service. This matters because cloud inference can increase network dependency, operating cost and privacy exposure, whereas local inference keeps raw images closer to the user. The technical challenge is that edge devices impose strict limits on model size, latency and energy use, so a high-accuracy model is not automatically the best deployment option.

CIFAR-10 (Krizhevsky, 2009) serves as the benchmark proxy: 60,000 RGB images at 32 × 32 pixels across ten balanced classes, deliberately chosen because its low resolution mirrors the degraded visual conditions of an edge camera at range. However, it does not reproduce the full complexity of real smart-home footage, such as night-time lighting, motion blur, weather effects, unusual camera angles, privacy-sensitive backgrounds or highly imbalanced event frequencies. The aim of this project is therefore not to deliver a production system, but to develop a reproducible deep learning prototype and critically evaluate whether this approach is feasible for future edge deployment — mapping the accuracy–efficiency Pareto frontier, exposing failure modes, and identifying the conditions under which a compact model could responsibly replace a human review loop.

---

## 2. Dataset and Preprocessing

CIFAR-10 was selected for its balanced class distribution (6,000 images per class), which eliminates class-prior bias during evaluation, and for its 32 × 32 resolution, which is directly analogous to the degraded output of a wide-angle, low-cost edge camera. The ten classes span domestic objects relevant to home monitoring — cats, dogs, birds — alongside vehicles and outdoor objects. CIFAR-10 vehicle categories are treated only as generic benchmark classes for evaluating visual discrimination. The practical recommendation is based on the accuracy–efficiency trade-off across models rather than on accuracy alone.

A 45k/5k/10k train/validation/test split was applied with seed 42; the test set was held out entirely and evaluated once per model to prevent optimistic bias. This three-way split supports more reliable model selection because validation performance guides design decisions and the test set remains an independent estimate of generalisation (Hastie et al., 2009).

Preprocessing normalised the images using CIFAR-10 channel statistics, with transformations applied only to the training split where appropriate. This reduces the risk of test-set leakage and stabilises gradient-based optimisation. The final training pipeline included random horizontal flips and random crops with padding. A 15-epoch ablation on CompactCNN showed a counterintuitive result: the no-augmentation version achieved 80.20% best validation accuracy, compared with 77.38% when augmentation was applied. This result is consistent with findings by Hernández-García and König (2018), who demonstrated that for compact architectures with limited parameter capacity, augmentation-induced data variety slows convergence within a fixed epoch budget. This does not prove augmentation is harmful in general; it more plausibly shows that for a compact model and short training window, added variation can slow convergence before its regularisation benefit is realised.

A critical limitation is that CIFAR-10 should not be treated as equivalent to real smart-home footage. It lacks the motion blur, lighting variation, camera angle bias, weather degradation, privacy-sensitive backgrounds, and severe class imbalance typical of production deployments where negative triggers (empty frames, domestic animals) may represent >99% of all captures. These gaps between laboratory benchmarks and deployment reality are directly addressed in the following sections.

---

## 3. Deep Learning Prototype and Design Decisions

Five deep learning architectures were evaluated to map the accuracy–efficiency trade-off for low-resolution edge image classification: one compact CNN trained from scratch, two lightweight pretrained CNNs, one deeper residual CNN, and one vision transformer. This allowed the project to compare genuinely edge-deployable models with larger architectures that establish a benchmark accuracy ceiling.

The main deployable model is CompactCNN, a custom three-block convolutional neural network with 620,810 parameters and a 2.48 MB footprint. CNNs are suitable for image classification because they learn local spatial patterns such as edges, textures and object parts directly from pixel neighbourhoods (LeCun et al., 1998). This is preferable to flattening images into independent numerical inputs, which discards important spatial structure. CompactCNN was therefore designed for edge feasibility rather than maximum benchmark accuracy. It uses progressively wider convolutional blocks, moving from 32 to 64 to 128 filters, with batch normalisation and dropout. Batch normalisation stabilises activations and supports faster convergence, while dropout reduces overfitting by randomly zeroing activations during training (Srivastava et al., 2014). The final model used dropout of 0.5 in the feature extractor and 0.3 in the classifier, and was trained for 60 epochs from scratch. It achieved 84.14% test accuracy with 0.69 ms average CPU inference latency in the notebook benchmark, making it the only model satisfying both the under-10 MB footprint and sub-millisecond latency target. However, this latency should be remeasured on the actual Raspberry Pi 4 before deployment, because host CPU benchmarking does not fully represent ARM hardware, thermal throttling or camera pipeline overhead.

The pretrained models provided useful comparators. MobileNetV3-Small has 1,528,106 parameters and a 6.11 MB footprint. It uses depthwise-separable convolutions and hardware-aware design choices to reduce computational cost (Howard et al., 2019). Despite ImageNet pre-training, it achieved 83.43%, slightly below CompactCNN. This is likely due to resolution mismatch: pretrained features are normally learned from 224 × 224 images, whereas CIFAR-10 images are only 32 × 32. Although MobileNetV3-Small fits the footprint budget, its 5.75 ms CPU latency means it misses the strict latency target.

ResNet-18 was included as a stronger CNN baseline. It uses residual skip connections to improve gradient flow in deeper networks (He et al., 2016). For CIFAR-10, the initial stride was changed to 1 and max-pooling was removed to prevent the small input from collapsing spatially too early. It was fine-tuned in two stages: first training only the new classifier head, then unfreezing the full backbone at a lower learning rate to reduce catastrophic forgetting. ResNet-18 achieved 94.94% accuracy, but its 44.73 MB size and 16.26 ms latency make it unsuitable for the target device.

EfficientNet-B0 applies compound scaling across depth, width and resolution (Tan and Le, 2019). It achieved 91.14% accuracy with fewer parameters than ResNet-18, showing strong parameter efficiency. However, its 16.08 MB footprint and 14.06 ms latency still exceed the project constraints. Swin-T was included to represent transformer-based vision models using shifted-window attention (Liu et al., 2021). It achieved the highest accuracy, 97.83%, but required upsampling to 224 × 224, resulting in a 110.11 MB footprint and 95.67 ms latency. It therefore demonstrates that state-of-the-art accuracy does not automatically translate into edge deployability.

Across all neural models, AdamW was used because it decouples weight decay from gradient updates (Loshchilov and Hutter, 2019), while cosine annealing reduced the learning rate smoothly during training (Loshchilov and Hutter, 2017). Overall, the results show that model selection must be based on deployment constraints, not accuracy alone. CompactCNN offers the strongest practical trade-off for the defined Raspberry Pi edge-AI scenario.

---

## 4. Traditional Machine Learning Comparison

A HOG + LinearSVC pipeline was implemented as the traditional machine learning baseline (Dalal and Triggs, 2005). HOG computes gradient orientation histograms across 4 × 4 pixel cells with 2 × 2 block normalisation, producing a fixed 1,764-dimensional descriptor per image. A LinearSVC (C = 0.1) then learns linear class boundaries in this hand-crafted feature space, achieving 40.85% test accuracy. Its main strength is computational simplicity: the model is small, fast and more transparent than a CNN. For very small microcontroller-class deployments, this type of approach may still be justified if moderate accuracy is acceptable.

The weakness is representational rigidity. HOG can describe local gradients, but it cannot adapt its feature hierarchy to distinguish fine-grained object differences in the same way as a trained CNN. At 32 × 32 resolution, visually similar classes such as cats and dogs or automobiles and trucks share broad shapes and textures. Handcrafted descriptors struggle because the engineer decides the feature representation in advance. By contrast, CompactCNN learns hierarchical features from data: early layers identify simple edges, intermediate layers combine them into textures, and deeper layers represent more class-specific patterns. The improvement from 40.85% to 84.14% demonstrates that deep learning is justified here by measurable performance gain, not novelty.

Table 1 contrasts the paradigms across deployment-relevant criteria.

**Table 1: Paradigm comparison across deployment-relevant criteria**

| Criterion | HOG + LinearSVC | CompactCNN (DL) |
|---|---|---|
| Feature extraction | Hand-crafted gradient histograms | Learned end-to-end via backpropagation |
| Spatial awareness | Sparse — 4 cells per dimension at 32×32 | Hierarchical — preserved through convolution |
| Training cost | < 1 min on CPU | ~20 min on RTX 3050 Ti GPU |
| Inference latency | < 1 ms (any device) | 0.69 ms (CPU, Raspberry Pi 4 feasible) |
| Test accuracy | 40.85% | 84.14% (+43.3 pp) |
| Model footprint | < 1 MB | 2.48 MB |
| Interpretability | Inherent — descriptor is human-readable | Requires post-hoc methods (GradCAM) |
| Edge suitability | Optimal for microcontrollers | Feasible on Pi 4; unsuitable below this tier |

HOG excels in two scenarios: microcontroller-class devices where floating-point CNNs are impractical, and interpretability-critical applications where feature transparency is a regulatory requirement. For the Raspberry Pi 4 target — which supports Python runtimes and PyTorch inference — the 43 percentage-point accuracy gap is operationally unacceptable for security-triage applications. Deep learning is justified not by novelty but by measurable performance improvement on a spatially structured task where hand-crafted features lack sufficient representational capacity. The comparison also shows a trade-off rather than a simple victory for deep learning. Traditional ML is easier to train, smaller and often more interpretable. Deep learning requires more training data, more computation and post-hoc explanation methods. Therefore, the correct conclusion is contextual: for low-resolution edge image classification on Raspberry Pi-class hardware, a compact CNN is justified because it substantially improves accuracy while remaining deployable; for more constrained devices, the HOG baseline remains a useful lower-complexity alternative.

---

## 5. Evaluation, Results, and Interpretability

Table 2 summarises the full accuracy–efficiency frontier. CompactCNN is the sole architecture satisfying both the ≤10 MB footprint and sub-millisecond latency constraints, making it the recommended deployment candidate. MobileNetV3-Small satisfies the footprint constraint but misses the latency target by 5×. ResNet-18 and EfficientNet-B0 are appropriate for less constrained edge hardware (e.g., NVIDIA Jetson Nano). Swin-T establishes the accuracy ceiling at 97.83% but is entirely impractical for real-time inference at 95.67 ms.

**Table 2: Accuracy–efficiency frontier across all evaluated architectures**

| Model | Test Acc. | Params | Size (MB) | FLOPs | CPU Latency | Edge Feasible? |
|---|---|---|---|---|---|---|
| HOG + LinearSVC | 40.85% | — | < 1 | — | < 1 ms | Yes (any device) |
| CompactCNN | 84.14% | 620,810 | 2.48 | 11.1M | 0.69 ms | ✓ (Pi 4) |
| MobileNetV3-Small | 83.43% | 1,528,106 | 6.11 | 6.0M | 5.75 ms | Partial (footprint ✓, latency ✗) |
| ResNet-18 | 94.94% | 11,181,642 | 44.73 | 565.8M | 16.26 ms | No (Pi 4) |
| EfficientNet-B0 | 91.14% | 4,020,358 | 16.08 | 34.4M | 14.06 ms | No (Pi 4) |
| Swin-T | 97.83% | 27,527,044 | 110.11 | 2,977M | 95.67 ms | No |

Grad-CAM was used to inspect ResNet-18 errors (Selvaraju et al., 2017). At 32 × 32 resolution, layer4 collapses to a 1 × 1 feature map, so layer3 was more informative. Failures showed background reliance in ship–airplane errors, fine-grained texture confusion in cat–dog cases, and structural similarity in automobile–truck errors. In deployment, these errors would not have equal cost; false alerts may reduce trust, while missed security-relevant events would require threshold calibration and human review.

---

## 6. Deployment, Ethical, Environmental, and Social Implications

The main deployment benefit of this prototype is privacy-preserving local inference. If images are processed directly on a Raspberry Pi 4, raw camera frames do not need to be transmitted to a third-party cloud service, reducing exposure to network interception, cloud retention and secondary use of household imagery. This aligns with the GDPR principles of purpose limitation and data minimisation, reflected in Article 5(1)(b), and with the UK Data Protection Act 2018. However, on-device processing is not automatically ethical. A smart-home classifier is dual-use: the same architecture could be retrained for person detection or intrusive monitoring. Responsible deployment would therefore require explicit purpose limitation, informed consent for people within the camera's field of view, short retention periods, local storage controls, and clear user-facing explanations of what the model can and cannot do.

Dataset bias and distributional shift are also major limitations. CIFAR-10 is balanced and curated, whereas real camera streams are usually imbalanced, repetitive and context-specific. In a home environment, most frames may contain no important event, while rare events are exactly the ones users care about. CIFAR-10 also reflects a limited visual context, so a model trained only on this benchmark may perform differently under varied lighting, weather, camera angles, backgrounds or regional object appearances. Thresholds would therefore need calibration on real, consent-based deployment footage. A human-in-the-loop review process should be used for low-confidence or security-relevant predictions, so the system supports decision-making rather than autonomous enforcement.

The environmental implication is mixed. Edge inference can reduce cloud data transfer and repeated server-side processing, but an always-on camera still consumes energy throughout its lifetime. Model choice therefore has cumulative impact. Swin-T achieved the highest accuracy but requires far more computation than CompactCNN, making it unsuitable for a low-power edge setting. Patterson et al. (2021) argue that energy use should be considered when evaluating machine-learning systems; this project supports that principle at a smaller scale. The responsible recommendation is to deploy the smallest model that meets the application's reliability requirements, benchmark it on physical Raspberry Pi 4 hardware, apply post-training quantisation where appropriate, monitor performance drift, and retrain only when real-world evidence justifies it.

---

## 7. Future Trends and Opportunities

**Hybrid mobile vision transformers.** MobileViT (Mehta and Rastegari, 2022) combines depthwise-separable convolutions with lightweight local self-attention blocks in a mobile-deployable architecture. MobileViT-XXS (1.3M parameters) achieves ImageNet-1K accuracy competitive with MobileNetV3-Large while capturing global scene context — presence of furniture, outdoor vegetation — that pure depthwise convolutions cannot encode. This is directly relevant to the failure modes identified in Section 5: ship→airplane confusions arise from the model attending to global context (sky/water) rather than object-level features. A future iteration of this system could adopt MobileViT-XXS as a Pareto-superior replacement for CompactCNN within the same 10 MB deployment budget, combining CompactCNN's latency efficiency with the global scene awareness that resolves contextual misclassification.

**Hardware-aware neural architecture search.** EfficientNet's NAS-derived compound scaling (Tan and Le, 2019) optimises architecture coefficients for ImageNet accuracy without hardware awareness. Once-for-All (OFA; Cai et al., 2020) addresses this directly: a single overparameterised network is trained from which sub-networks can be extracted and specialised for any target hardware without retraining. Applied to this project, OFA would generate a CIFAR-10 classifier optimised specifically for the Raspberry Pi 4's ARM Cortex-A72 — matching an exact latency and memory budget rather than approximating it through manual architecture search. This is significant because the stride adaptations and resolution-induced spatial collapse observed in Section 3 are artefacts of architectures designed for 224 × 224 inputs; a hardware-aware search would natively optimise for 32 × 32, eliminating these ad-hoc fixes and potentially recovering the accuracy gap between MobileNetV3-Small and CompactCNN.

**Foundation models and zero-shot edge classification.** CLIP (Radford et al., 2021) learns joint image–text representations at web scale, enabling zero-shot classification via text prompts without task-specific training. A smart-home classifier built on a CLIP backbone could be updated to new object categories through prompt engineering alone — no labelled retraining required. The current barrier is inference cost: CLIP-ViT-B/32 requires ~150× the FLOPs of CompactCNN per image. Lightweight successors such as SigLIP (Zhai et al., 2023) reduce this gap substantially. Within the five-year deployment window of a smart-home product, prompt-based edge classification may eliminate the need for labelled retraining datasets entirely — directly addressing the dataset bias and distributional shift limitations identified in Section 6, where collecting culturally representative training data at scale is both expensive and privacy-sensitive.

---

## 8. AI Assistance and Conclusion

AI assistance was used as a collaborative support tool for idea development, report structuring, debugging guidance and explanation of model-design options. It was not used as a substitute for understanding the implementation. All AI-generated code was empirically tested and per-architecture validated — the stride fix, for example, required separate implementation for each backbone because access patterns differ between architectures. No AI-generated text appears verbatim in this report or the notebook. The GAIT framework was followed throughout; this declaration is consistent with the AI Collaboration scale specified in the assessment brief.

Overall, this project demonstrates that a compact CNN can serve as a viable proof-of-concept for low-resolution, privacy-preserving edge image classification. Deep learning is justified because it substantially outperforms the traditional HOG + LinearSVC baseline while retaining practical deployability. CompactCNN achieves 84.14% accuracy at 0.69 ms CPU latency within a 2.48 MB footprint — the sole architecture satisfying both hard deployment constraints — while Swin-T establishes that transformer-class accuracy (97.83%) remains out of reach for sub-watt edge inference at current efficiency levels. The gap between benchmark performance and production readiness is substantial: distributional shift, class imbalance, threshold calibration, and quantisation are prerequisites not addressed in this study. CIFAR-10 remains a benchmark proxy; responsible deployment would require real-device latency testing, real-camera validation, quantisation, bias assessment, privacy safeguards and ongoing monitoring. The primary value of the prototype is therefore diagnostic — it identifies where the accuracy–efficiency frontier lies, which failure modes are structurally attributable to resolution limitations, and which emerging architectures (MobileViT, OFA, SigLIP) offer the most credible path from laboratory benchmark to responsible deployment.

---

## References

Cai, H., Gan, C., Wang, T., Zhang, Z. and Han, S. (2020) 'Once-for-All: Train one network and specialize it for efficient deployment', in *Proceedings of the 8th International Conference on Learning Representations (ICLR)*. Available at: https://arxiv.org/abs/1908.09791 (Accessed: 5 May 2026).

Dalal, N. and Triggs, B. (2005) 'Histograms of oriented gradients for human detection', in *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, San Diego, CA, June 2005, vol. 1, pp. 886–893.

Hastie, T., Tibshirani, R. and Friedman, J. (2009) *The elements of statistical learning: data mining, inference, and prediction*. 2nd edn. New York: Springer.

He, K., Zhang, X., Ren, S. and Sun, J. (2016) 'Deep residual learning for image recognition', in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, June 2016, pp. 770–778.

Hernández-García, A. and König, P. (2018) 'Data augmentation instead of explicit regularization', *arXiv preprint* arXiv:1806.03852. Available at: https://arxiv.org/abs/1806.03852 (Accessed: 5 May 2026).

Howard, A. et al. (2019) 'Searching for MobileNetV3', in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, Seoul, October 2019, pp. 1314–1324.

Krizhevsky, A. (2009) *Learning multiple layers of features from tiny images*. Technical report. University of Toronto.

LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P. (1998) 'Gradient-based learning applied to document recognition', *Proceedings of the IEEE*, 86(11), pp. 2278–2324.

Liu, Z. et al. (2021) 'Swin Transformer: Hierarchical vision transformer using shifted windows', in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, October 2021, pp. 10012–10022.

Loshchilov, I. and Hutter, F. (2017) 'SGDR: Stochastic gradient descent with warm restarts', in *Proceedings of the 5th International Conference on Learning Representations (ICLR)*, Toulon, April 2017. Available at: https://arxiv.org/abs/1608.03983 (Accessed: 5 May 2026).

Loshchilov, I. and Hutter, F. (2019) 'Decoupled weight decay regularization', in *Proceedings of the 7th International Conference on Learning Representations (ICLR)*, New Orleans, LA, May 2019. Available at: https://arxiv.org/abs/1711.05101 (Accessed: 5 May 2026).

Mehta, S. and Rastegari, M. (2022) 'MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer', in *Proceedings of the 10th International Conference on Learning Representations (ICLR)*, April 2022. Available at: https://arxiv.org/abs/2110.02178 (Accessed: 5 May 2026).

Patterson, D. et al. (2021) 'Carbon emissions and large neural network training', *arXiv preprint* arXiv:2104.10350. Available at: https://arxiv.org/abs/2104.10350 (Accessed: 5 May 2026).

Radford, A. et al. (2021) 'Learning transferable visual models from natural language supervision', in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, July 2021, pp. 8748–8763.

Selvaraju, R.R. et al. (2017) 'Grad-CAM: Visual explanations from deep networks via gradient-based localization', in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, Venice, October 2017, pp. 618–626.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R. (2014) 'Dropout: A simple way to prevent neural networks from overfitting', *Journal of Machine Learning Research*, 15(1), pp. 1929–1958.

Tan, M. and Le, Q.V. (2019) 'EfficientNet: Rethinking model scaling for convolutional neural networks', in *Proceedings of the 36th International Conference on Machine Learning (ICML)*, Long Beach, CA, June 2019, pp. 6105–6114.

Zhai, X. et al. (2023) 'Sigmoid loss for language image pre-training', in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, Paris, October 2023. Available at: https://arxiv.org/abs/2303.15343 (Accessed: 5 May 2026).

---

## Appendix A — Extended GradCAM Visualisations

Full Grad-CAM heatmaps for both correct predictions and failure cases across `layer3` and `layer4` of ResNet-18 are available in the accompanying notebook outputs (`gradcam_correct_l3.png`, `gradcam_correct_l4.png`, `gradcam_failures_l3.png`, `gradcam_failures_l4.png`). These exceed what can be meaningfully reproduced at 32 × 32 pixel resolution within a text figure.

## Appendix B — AI Assistance Log

AI tool used: Claude (Anthropic). Primary uses: PyTorch training loop boilerplate, GradCAM hook registration pattern, CIFAR-10 stride adaptation debugging, report structure suggestions. All outputs reviewed, empirically tested and adapted per-architecture before inclusion. Declared in accordance with GAIT framework and WM9B7-15 AI Collaboration policy.
