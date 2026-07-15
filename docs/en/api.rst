API Reference
=============

BaoIAD is built on `MMEngine <https://mmengine.readthedocs.io/en/latest/>`_ and follows its registry-based architecture.
Every component — model, dataset, transform, metric, hook — is registered into a global registry and can be looked up by name in config files.

.. tip:: For tutorials on adding custom components, see :doc:`advanced_guides/index`.

This page documents the public API organized by subsystem.


Registries (``baoiad.registry``)
--------------------------------

All registries inherit from their MMEngine counterparts under the ``baoiad`` scope.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Registry
     - Purpose
   * - ``MODELS``
     - Detectors, backbones, necks, heads, losses
   * - ``DATASETS``
     - Dataset classes for each benchmark
   * - ``TRANSFORMS``
     - Data loading, augmentation, and packing transforms
   * - ``METRICS``
     - Evaluation metrics (AUROC, AUPRO, AUPIMO, …)
   * - ``HOOKS``
     - Training hooks (memory bank, strict-mode, visualization)
   * - ``LOOPS``
     - Custom training/test loops
   * - ``RUNNERS``
     - Runner implementations
   * - ``VISUALIZERS``
     - Visualization backends
   * - ``DATA_SAMPLERS``
     - Data samplers for dataset iteration
   * - ``OPTIMIZERS``
     - Optimizer constructors
   * - ``OPTIM_WRAPPERS``
     - Optimizer wrappers
   * - ``OPTIM_WRAPPER_CONSTRUCTORS``
     - Optimizer wrapper constructor functions
   * - ``PARAM_SCHEDULERS``
     - Learning-rate and momentum schedulers

Usage example::

   from baoiad.registry import MODELS

   # Look up a registered detector
   model_cls = MODELS.get('PatchCore')


Models (``baoiad.models``)
--------------------------

Base Classes
~~~~~~~~~~~~

All detectors inherit from :class:`~baoiad.models.base_ad_model.BaseADModel`.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``BaseADModel``
     - Abstract base for all anomaly detection models
   * - ``MemoryBankADModel``
     - Feature-memory methods (PatchCore, PaDiM, …)
   * - ``KnowledgeDistillationADModel``
     - Teacher-student methods (RD, EfficientAD, …)
   * - ``FlowBasedADModel``
     - Normalizing-flow methods (FastFlow, CFlow-AD, …)
   * - ``ReconstructionADModel``
     - Reconstruction-based methods (AutoEncoder, …)
   * - ``VisionLanguageADModel``
     - Vision-language methods (WinCLIP, AnomalyCLIP, …)
   * - ``DiscriminatorADModel``
     - Discriminative methods (SimpleNet, …)

Detectors (``baoiad.models.detectors``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

37 detector implementations spanning 9 families. Key examples:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Class
     - Family
     - Reference
   * - ``PatchCore``
     - Feature Memory
     - Roth et al., CVPR 2022
   * - ``PaDiM``
     - Feature Memory
     - Defard et al., ICPR 2021
   * - ``EfficientAD``
     - Knowledge Distillation
     - Batzner et al., WACV 2024
   * - ``ReverseDistillation``
     - Knowledge Distillation
     - Deng & Li, CVPR 2022
   * - ``FastFlow``
     - Normalizing Flow
     - Yu et al., arXiv 2021
   * - ``CFlowAD``
     - Normalizing Flow
     - Gudovskiy et al., WACV 2022
   * - ``DRAEM``
     - Self-supervised Synthesis
     - Zavrtanik et al., ICCV 2021
   * - ``SimpleNet``
     - Discriminative
     - Liu et al., CVPR 2023
   * - ``WinCLIP``
     - Vision-Language
     - Jeong et al., CVPR 2023
   * - ``Dinomaly``
     - Hybrid/Unified
     - Guo et al., 2024

See :doc:`model_zoo` for the full list and benchmark results.

Backbones (``baoiad.models.backbones``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-trained feature extractors wrapped for the BaoIAD registry:

- ``TIMMBackbone`` — Any backbone from `timm <https://huggingface.co/docs/timm>`_.
- ``TorchvisionBackbone`` — Backbones from ``torchvision.models``.
- ``OpenCLIPBackbone`` — Backbones from `OpenCLIP <https://github.com/mlfoundations/open_clip>`_.

Necks (``baoiad.models.necks``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature-processing modules between backbone and head:

- ``MultiFeatureNeck`` — Extracts features from multiple backbone layers.

Heads (``baoiad.models.heads``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Task-specific output heads:

- ``AnomalyMapHead`` — Generates pixel-level anomaly maps.
- ``MemoryBankHead`` — Feature bank construction and scoring.

Losses (``baoiad.models.losses``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loss functions registered with the ``MODELS`` registry (MMEngine convention):

- ``FocalLoss``, ``SSIMLoss``, ``L2Loss``, ``CosineLoss``


Datasets & Transforms
---------------------

Datasets (``baoiad.datasets``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All datasets inherit from :class:`~baoiad.datasets.base_ad_dataset.BaseADDataset`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``BaseADDataset``
     - Base class with shared loading logic for all benchmarks
   * - ``MVTecADDataset``
     - MVTec AD (15 texture/object categories)
   * - ``VisADataset``
     - VisA (12 object categories)
   * - ``BTechDataset``
     - BTech (3 categories)
   * - ``MVTec3DDataset``
     - MVTec 3D-AD (RGB + 3D point cloud)
   * - ``MVTecLOCODataset``
     - MVTec LOCO (logical constraints)
   * - ``MPDDDataset``
     - MPDD (metallic defects)
   * - ``MVTecAD2Dataset``
     - MVTec AD 2 (next-generation MVTec)
   * - ``KolektorDataset``
     - Kolektor surface-defect dataset
   * - ``VADDataset``
     - VAD anomaly detection dataset
   * - ``RealIADDataset``
     - Real-IAD (multi-view, high-resolution)
   * - ``RealIADD3Dataset``
     - Real-IAD D³ (RGB + 3D + P3D modalities)
   * - ``DRAEMDataset``
     - DRAEM self-supervised synthesis dataset
   * - ``NSATrainDataset``
     - NSA training dataset (synthetic anomaly generation)
   * - ``GLASSDataset``
     - GLASS dataset
   * - ``RegADTrainDataset``
     - RegAD few-shot training dataset
   * - ``RegADTestDataset``
     - RegAD few-shot test dataset
   * - ``ClinicDBDataset``
     - Medical: ClinicDB
   * - ``ColonDBDataset``
     - Medical: ColonDB
   * - ``AdaCLIPClinicDBDataset``
     - AdaCLIP variant of ClinicDB
   * - ``AdaCLIPColonDBDataset``
     - AdaCLIP variant of ColonDB
   * - ``AdaCLIPVisADataset``
     - AdaCLIP variant of VisA
   * - ``AACLIPJsonDataset``
     - AACLIP JSON-based dataset

See :doc:`user_guides/prepare_dataset` for dataset setup instructions.

Samplers (``baoiad.datasets.samplers``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom samplers for controlling data iteration order:

- ``PersistentShuffleSampler`` — Shuffle with deterministic seed persistence across runs.
- ``OpenIADSubsetRandomSampler`` — Subset sampling for OpenIAD-compatible datasets.
- ``PythonShuffleSampler`` — Pure-Python shuffle sampler (avoids torch workers).
- ``ExplicitOrderSampler`` — Deterministic fixed-order sampler.
- ``PerEpochOrderSampler`` — Per-epoch reordering sampler.

Transforms (``baoiad.datasets.transforms``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loading
.......

- ``LoadImage`` — Load an image from disk.
- ``LoadMask`` — Load a pixel-level ground-truth mask.

Augmentation
............

- ``ResizeAD`` — Resize image (and optional mask).
- ``RandomRotation`` — Random rotation augmentation.
- ``NormalizeAD`` — Normalize with per-channel mean/std.
- ``ScaleNormalizeAD`` — Scale-based normalization.
- ``OpenCLIPPreprocessAD`` — Preprocessing matching OpenCLIP's expected input.
- ``CenterCrop`` — Center crop.
- ``ThresholdMask`` — Binarize masks at a given threshold.

Method-Specific
...............

- ``CFlowOfficialTransform`` — CFlow-AD official training transform pipeline.
- ``DeSTSegAugment`` — DeSTSeg augmentation pipeline.
- ``PackDeSTSegInputs`` — Pack inputs for DeSTSeg.

Formatting
..........

- ``PackADInputs`` — Collate image, mask, and meta into MMEngine data samples.
- ``PackDRAEMInputs`` — Collate inputs for DRAEM-style models.


Evaluation (``baoiad.evaluation``)
----------------------------------

Metric Classes
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``AnomalyDetectionMetric``
     - Main metric: computes image-level AUROC, F1-max, AP, AUPRO, AUPIMO, ECE, FPR@95TPR
   * - ``AACLIPOfficialMetric``
     - AACLIP official evaluation metric
   * - ``AnomalyMapMeanMetric``
     - Mean anomaly-map score metric

Standalone Functions
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - ``compute_aupro``
     - Compute per-instance AUPRO
   * - ``compute_pimo``
     - Compute per-instance AUPIMO
   * - ``compute_ece``
     - Compute image-level Expected Calibration Error
   * - ``compute_pixel_ece``
     - Compute pixel-level Expected Calibration Error
   * - ``compute_fpr_at_tpr``
     - Compute FPR at a given TPR threshold (default 95%)
   * - ``measure_speed``
     - Measure inference throughput (FPS / latency)

See :doc:`user_guides/evaluate_results` for interpreting evaluation results.


Visualization (``baoiad.visualization``)
----------------------------------------

- ``ADVisualizer`` — Renders anomaly heatmaps, overlays, and side-by-side comparisons. Registered in ``VISUALIZERS``.
- ``ADVisualizationHook`` — Hook that triggers visualization during testing. Registered in ``HOOKS``.

See :doc:`user_guides/visualization` for visualization usage.


Hooks & Loops (``baoiad.engine``)
---------------------------------

Hooks (``baoiad.engine.hooks``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``MemoryBankHook``
     - Manages feature-memory bank construction during training
   * - ``MemSegStrictTrainHook``
     - Enforces MemSeg strict training protocol
   * - ``UFlowStrictTrainHook``
     - Enforces UFlow strict training protocol
   * - ``ViTADStrictTrainHook``
     - Enforces ViTAD strict training protocol
   * - ``ADVisualizationHook``
     - Generates anomaly visualizations during testing

See :doc:`advanced_guides/add_custom_hook` for writing custom hooks.


Structures (``baoiad.structures``)
----------------------------------

- ``ADDataSample`` — Data structure carrying image-level label, anomaly score, pixel-level anomaly map, and metadata through the pipeline. Based on MMEngine's ``BaseDataElement``.


Utilities
---------

Environment
~~~~~~~~~~~

- ``BAOIAD_DATA_ROOT`` — Top-level package variable (``str``). Resolved from the ``BAOIAD_DATA_ROOT`` environment variable, falling back to ``data/`` relative to the repo root.
- ``HF_ENDPOINT`` — Optional Hugging Face endpoint selected explicitly by the user or deployment environment. BaoIAD does not set or rewrite it implicitly.
