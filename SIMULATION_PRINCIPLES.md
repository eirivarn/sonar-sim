# Sonar Simulation: Principles and Methods

## Overview

This document describes the principles and methodology of the synthetic sonar data generation system developed for training and evaluating underwater net detection algorithms. The simulation framework is designed to produce realistic Forward-Looking Sonar (FLS) imagery with accurate acoustic physics modeling and pixel-perfect ground truth segmentation labels.

## 1. Simulation Architecture

### 1.1 Voxel-Based World Representation

The simulation uses a discrete 2D voxel grid representation of the underwater environment. Each voxel (volume element) in the grid contains:

- **Material Type**: Defines the acoustic properties (e.g., water, net, fish, rock)
- **Spatial Position**: 2D coordinates in the world frame
- **Acoustic Properties**: Reflectivity, absorption, and scattering characteristics

This discrete representation enables:
- Efficient ray-marching algorithms for sonar beam propagation
- Deterministic ground truth generation
- Precise material identification at every pixel
- Fast simulation suitable for large-scale dataset generation

### 1.2 Coordinate Systems

The simulation employs two primary coordinate frames:

**Polar Coordinates (Sensor Native)**
- Range bins: Discrete distances from the sonar transducer (typically 1024 bins)
- Azimuth beams: Angular sectors covering the field of view (typically 256 beams)
- Represents the raw sonar acquisition geometry

**Cartesian Coordinates (World Frame)**
- Standard XY coordinate system for scene representation
- Used for visualization and spatial reasoning
- Conversion between frames uses precise geometric transformations

## 2. Acoustic Physics Modeling

### 2.1 Core Physical Phenomena

The simulation implements several key acoustic phenomena observed in real sonar systems:

#### Water Attenuation
Sound waves lose energy as they propagate through water due to:
- **Absorption**: Energy converted to heat (frequency-dependent)
- **Geometric spreading**: Energy distributed over increasing area with distance
- **Implementation**: Range-dependent intensity decay using exponential attenuation model

#### Acoustic Speckle
Coherent interference patterns caused by:
- Multiple scattering from sub-resolution structures
- Phase variations in reflected wavefronts
- **Characteristics**: Multiplicative noise with Gamma distribution
- **Implementation**: Spatially correlated speckle patterns with configurable shape and aspect ratio

#### Angular Scattering
Target reflectivity depends on:
- **Angle of incidence**: Specular vs. diffuse reflection
- **Surface orientation**: Normal angles relative to beam direction
- **Implementation**: Angle-dependent gain with configurable falloff characteristics

#### Multi-Path Propagation
Complex acoustic paths including:
- **Direct path**: Primary return from target
- **Surface/bottom bounce**: Secondary reflections
- **Multiple scattering**: Inter-object reflections

#### Shadowing and Occlusion
Acoustic shadows occur when:
- Dense objects block sound propagation
- Material absorption prevents beam penetration
- **Implementation**: Ray-marching detects occlusions and reduces intensity in shadow regions

### 2.2 Material-Specific Properties

Different materials exhibit distinct acoustic behaviors:

**Fish Cage Nets**
- High reflectivity due to material impedance mismatch with water
- Thin structure causes partial transmission
- Angular dependency: Strong returns at perpendicular incidence
- Scattering: Both specular and diffuse components

**Fish**
- Moderate reflectivity from swim bladder (gas/water interface)
- Size-dependent scattering (Rayleigh to geometric regimes)
- Motion creates temporal decorrelation
- Density-dependent intensity variations

**Water Column**
- Low baseline reflectivity
- Volume scattering from particulates and organisms
- Range-dependent backscatter intensity

**Boundaries**
- Surface: Strong reflection with roughness-dependent scattering
- Bottom: Sediment-dependent reflectivity
- Side boundaries: Typically absorbing in confined environments

### 2.3 Configurable Physics Parameters

The simulation provides extensive control over acoustic phenomena through 26+ configurable parameters:

**Attenuation Controls**
- Water absorption coefficient
- Range-dependent decay power law
- Frequency-dependent absorption factor

**Speckle Parameters**
- Intensity (magnitude of speckle noise)
- Spatial scale (correlation length)
- Shape parameter (Gamma distribution)
- Aspect ratio variation (anisotropic speckle)

**Scattering Effects**
- Angle-dependent scatter strength
- Edge scattering enhancement
- Grouped scatter patches (localized high-reflectivity regions)
- Density-dependent scatter boost

**Noise and Artifacts**
- Gaussian sensor noise
- Range jitter (bin misregistration)
- Azimuth streaks (saturation artifacts)
- Temporal decorrelation (frame-to-frame variations)

**Beam Pattern Effects**
- Beam spreading (resolution degradation with range)
- Cross-range blur (azimuth resolution)
- Multi-bin spreading (range resolution)

## 3. Sonar Sensor Configuration

### 3.1 Sensor Parameters

The simulation models a Forward-Looking Sonar with the following configurable parameters:

**Geometric Parameters**
- **Field of View (FOV)**: Angular coverage (e.g., 120°)
- **Maximum Range**: Detection distance (e.g., 20 meters)
- **Resolution**: 
  - Range bins: Radial sampling density (e.g., 1024 bins)
  - Azimuth beams: Angular sampling density (e.g., 256 beams)

**Acoustic Parameters**
- **Operating Frequency**: Affects attenuation and resolution (typically 1-2 MHz for imaging sonars)
- **Pulse Duration**: Determines range resolution
- **Beam Width**: Controls angular resolution

**Platform Parameters**
- **Position**: 2D coordinates in world frame
- **Orientation**: Heading angle
- **Velocity**: For Doppler effects (optional)

### 3.2 Real Sensor Correspondence

The simulation is designed to match the characteristics of commercial imaging sonars such as:
- Tritech Gemini series
- Blueprint Oculus series
- Teledyne BlueView series

**Key matching characteristics:**
1. **Polar acquisition geometry**: Native sensor coordinates preserved
2. **Resolution characteristics**: Realistic range-azimuth resolution trade-offs
3. **Dynamic range**: 8-bit or 16-bit intensity quantization
4. **Noise characteristics**: Matched speckle and sensor noise statistics
5. **Artifact patterns**: Realistic azimuth streaks, shadows, and multi-path returns

### 3.3 Image Formation Process

The sonar image generation follows these steps:

1. **Ray Casting**: For each beam angle:
   - Cast ray from sonar origin at specified angle
   - March along ray in discrete steps (sub-voxel sampling)
   - Accumulate interactions with materials

2. **Range Sampling**: For each range bin:
   - Integrate backscatter contributions within range gate
   - Apply material-specific reflectivity
   - Account for angle of incidence

3. **Physics Application**: Apply acoustic effects:
   - Range-dependent attenuation
   - Angular scattering modulation
   - Speckle noise overlay
   - Shadow casting
   - Multi-path contributions

4. **Sensor Effects**: Apply detection chain artifacts:
   - Quantization (bit depth)
   - Sensor noise
   - Saturation (dynamic range limits)
   - Time-varying gain (TVG) curves

5. **Output Formation**: Generate final image:
   - Polar format: (range_bins × azimuth_beams)
   - Cartesian projection (optional): Geometric transformation to XY grid
   - Intensity normalization and scaling

## 4. Ground Truth Generation

### 4.1 Semantic Segmentation Labels

Ground truth is generated simultaneously with sonar imagery, ensuring perfect spatial alignment:

**Label Generation Process**
1. During ray-marching, record material ID at each voxel intersection
2. Create segmentation map with same dimensions as sonar image
3. Assign class labels based on material type:
   - Class 0: Water/background
   - Class 1: Net structures
   - Class 2: Fish (optional)
   - Class N: Additional object classes

**Key Properties**
- **Pixel-perfect alignment**: Ground truth matches sonar geometry exactly
- **No annotation error**: Labels are deterministically generated from scene geometry
- **Multi-class support**: Arbitrary number of material classes
- **Temporal consistency**: Labels track objects across frames

### 4.2 Label Representations

Ground truth is provided in multiple formats:

**Binary Masks**
- Single-class segmentation (e.g., net vs. background)
- Uint8 format: 0 (background) or 255 (target)
- Suitable for binary segmentation models

**Multi-class Masks**
- Integer class IDs for each pixel
- Supports arbitrary number of classes
- Enables multi-task learning scenarios

**Instance Segmentation** (optional)
- Separate object instances with unique IDs
- Enables tracking and counting applications

### 4.3 Coordinate Frame Correspondence

Ground truth is available in both coordinate systems:

**Polar Ground Truth**
- Native sensor coordinates (range × azimuth)
- Matches raw sonar data geometry
- Suitable for end-to-end learning on polar data

**Cartesian Ground Truth**
- World-frame coordinates (X × Y)
- Easier for spatial reasoning and visualization
- Requires same polar-to-Cartesian transform as sonar imagery

### 4.4 Metadata and Annotations

Each frame includes comprehensive metadata:
- **Timestamp**: Unix nanosecond timestamp for temporal ordering
- **Sensor pose**: Position and orientation in world frame
- **Spatial extent**: Bounding box in world coordinates
- **Scene configuration**: Material properties and object parameters
- **Physics settings**: Active acoustic parameters for reproducibility

## 5. Scene Generation and World Diversity

### 5.1 Scene Architecture

Scenes are defined by three core components:

**Geometry Definition**
- Primitive shapes: Lines, circles, rectangles, polygons
- Material assignment: Each primitive has associated material properties
- Hierarchical composition: Complex structures from simple primitives

**Dynamic Objects**
- Autonomous agents with behavioral models (e.g., fish swimming patterns)
- Configurable movement: Velocities, accelerations, boundary interactions
- Temporal evolution: Objects move and interact over time

**Environment Parameters**
- World boundaries and dimensions
- Ambient conditions (water properties, particulate density)
- Light/visibility (for visualization, not acoustic simulation)

### 5.2 Implemented Scene Types

**Fish Cage Scene (Aquaculture)**
- Cylindrical or rectangular net enclosures
- Variable cage dimensions and net mesh density
- Fish population with schooling behavior
- Seabed and surface boundaries
- Realistic for underwater aquaculture inspection tasks

**Urban Scene (Subsurface Inspection)**
- Underground infrastructure (pipes, tunnels)
- Debris and obstacles
- Moving elements (vehicles, people)
- Suitable for urban sonar applications

### 5.3 Procedural Generation Capabilities

The framework supports automated scene generation:

**Parametric Variation**
- Systematic parameter sweeps for dataset diversity
- Random sampling from parameter distributions
- Ensures coverage of operational envelope

**Geometric Randomization**
- Net topology: Straight, curved, slack, taut configurations
- Object placement: Stochastic positioning within constraints
- Scale variation: Different sizes for same object types

**Environmental Conditions**
- Variable water properties (temperature, salinity affecting attenuation)
- Particulate density (affecting volume scattering)
- Surface conditions (rough vs. calm)

### 5.4 Data Collection Modes

**Fixed-Path Collection**
- **Circular paths**: Orbit around target at fixed radius
- **Grid patterns**: Systematic spatial coverage
- **Spiral trajectories**: Gradual approach to target
- Configurable: Radius, spacing, orientation modes

**Random Exploration**
- Robot dynamics simulation: Realistic motion constraints
- Collision avoidance: Stay within safe boundaries
- Randomized trajectories: Maximum viewpoint diversity
- Suitable for training robust models

**Interactive Control**
- Manual positioning for specific scenarios
- Real-time scene manipulation
- Useful for edge case generation

## 6. Dataset Generation Pipeline

### 6.1 Physics-Based Augmentation

To maximize dataset diversity without re-running expensive simulations, physics parameter augmentation is employed:

**Strategy**
1. Generate base simulation with standard physics settings
2. Apply image-level approximations of different acoustic conditions
3. Create multiple physics "configurations" from same geometry
4. Result: N×M dataset (N simulation runs × M physics configs)

**Augmentation Categories**
- Attenuation variations: Simulate different water conditions
- Speckle variations: Model different frequencies and sea states
- Scattering variations: Represent different target materials
- Noise variations: Account for sensor and environmental variability

**Advantages**
- Computationally efficient: Image processing vs. full simulation
- Physics-informed: Based on acoustic principles, not arbitrary transforms
- Large-scale generation: 100,000+ samples feasible
- Controlled diversity: Systematic exploration of parameter space

### 6.2 Train/Validation Splitting

**Temporal Separation**
- Training and validation data from different simulation runs
- Ensures test set evaluates generalization, not memorization
- Prevents temporal correlation artifacts

**Physics Diversity**
- Different physics configurations in train vs. validation
- Tests robustness to acoustic condition variations
- Realistic assessment of real-world deployment performance

## 7. Applications and Use Cases

### 7.1 Supervised Learning

**Semantic Segmentation**
- Pixel-wise classification of net structures
- End-to-end learning from sonar to segmentation
- Handles complex backgrounds and occlusions

**Temporal Modeling**
- Video sequence processing with LSTM/ConvLSTM
- Exploit temporal consistency for improved accuracy
- Reject transient false positives

### 7.2 Sim-to-Real Transfer

**Domain Adaptation Strategies**
- Physics diversity mimics real-world variability
- Augmentation bridges sim-real domain gap
- Fine-tuning on small real datasets

**Evaluation Metrics**
- Intersection-over-Union (IoU) for segmentation quality
- Precision/Recall for detection tasks
- Temporal consistency metrics for video processing

### 7.3 Algorithm Development

**Benchmarking**
- Consistent test conditions across algorithms
- Quantitative performance comparison
- Ablation studies on physics parameters

**Failure Mode Analysis**
- Systematic testing of edge cases
- Controlled degradation (noise, blur, occlusion)
- Identify and address algorithm weaknesses

## 8. Validation and Realism

### 8.1 Physical Accuracy

The simulation implements established acoustic physics models:
- Sonar equation for range-dependent intensity
- Rayleigh and geometric scattering regimes
- Coherent and incoherent noise models
- Validated against theoretical predictions

### 8.2 Visual Realism

Qualitative assessment shows:
- Characteristic speckle patterns similar to real sonar
- Realistic shadow artifacts and occlusions
- Appropriate dynamic range and contrast
- Representative noise and artifact patterns

### 8.3 Statistical Validation

Quantitative metrics comparing simulated and real data:
- Intensity distribution histograms
- Spatial autocorrelation functions (speckle statistics)
- Signal-to-noise ratio profiles
- Frequency spectrum analysis

## 9. Computational Considerations

### 9.1 Performance Characteristics

**Simulation Speed**
- Typical throughput: 10-100 frames/second (depends on scene complexity)
- Headless operation: No visualization overhead
- Suitable for large-scale dataset generation (100k+ samples)

**Scalability**
- Voxel grid size: Adjustable for speed-accuracy trade-off
- Ray marching resolution: Configurable sampling density
- Physics complexity: Enable/disable effects as needed

### 9.2 Storage and Format

**Data Format**
- NPZ (NumPy compressed): Native simulation output
- PNG: Augmented training data (images and masks)
- JSON: Metadata and configuration

**Storage Requirements**
- Raw simulation: ~1-5 MB per frame (NPZ)
- Augmented training: ~100-500 KB per sample (PNG)
- 100,000 sample dataset: ~10-50 GB

## 10. Conclusion

This sonar simulation framework provides a comprehensive platform for generating realistic synthetic sonar data with accurate ground truth labels. The combination of physics-based acoustic modeling, flexible scene generation, and systematic dataset augmentation enables large-scale training dataset creation for machine learning applications in underwater perception.

Key strengths include:
- **Physical fidelity**: Realistic acoustic phenomena modeling
- **Ground truth accuracy**: Pixel-perfect semantic labels
- **Flexibility**: Extensive configurability and scene diversity
- **Scalability**: Efficient generation of large datasets
- **Reproducibility**: Deterministic simulation with full parameter logging

This synthetic data generation capability is essential for developing and evaluating robust sonar-based perception systems, particularly for applications where real data collection is expensive, dangerous, or impractical.
