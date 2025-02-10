<h1>Half-Object Acquisition and Tomographic Pre-Reconstruction Processing of Large Object</h1>
<p>
X-ray microtomography is a widely used imaging technique with a limited field of view, thus, scanning large objects requires employing an appropriate data acquisition scheme or stitching technique to obtain the entire object’s reconstruction. Half-object acquisition is a popular method for addressing such a scenario, where truncated data collected for a 360-degree rotating object is processed, yielding a representation of the entire object for 180-degree rotation. Whereas pre-reconstruction stitching of sinograms and post-reconstruction tomogram stitching techniques are well established, the current approach for processing radiographs exhibits limitations, especially when high-quality outcomes are demanded. In our study, we present and overcome those limitations with proposed reliable processing approach which can be generalized to other tomographic modalities. The proposed approach is demonstrated using microtomographic experimental data, yielding artifact-free tomograms with enhanced sharpness even for challenging datasets.
</p>
<h3>Paper:</h3>
<p>N.N. Shubayev, T. Mass, L. Broche, P. Zaslansky, "On Half-Object Acquisition and Tomographic Pre-Reconstruction Processing of Large Object: Limitations and a Reliable Approach", 2024</p> 
<h2>Pre-Reconstruction Processing Pipeline for Object Exceeding The Field of View:</h2>
<p>1. Obtain the data for 360-deg rotation of the object : radiographs, flat-fields (before and after scannig the object), dark fields </p>
<p>2. Perform Dynamic Flat-Field Correction (FFC) </p>
<img src="images\DFFC.png" alt="DFFC" width="500"/><br>
<p>3. Split the normalized radiographs into 2 stacks, each representing 180-deg rotation of the object</p>
<p>4. Perform histogram matching of coresponding radiographs</p>
<p>5. Flip horizontally one of the stacks </p>
<p>6. Stitch corresponding radiograph using Log-Polar Phase Correlation</p>
<img src="images\stitch.png" alt="DFFC" width="600"/><br>
<p>7. Reconstruct stitched radiographs</p>
<img src="images\image__rec1193.jpg" alt="DFFC" width="200"/><br>
<h2>Disclaimer</h2>
<h3>This code is currently in its preliminary stages, and significant updates are anticipated soon, including modifications, optimizations, and GPU support. In the meantime, the code should be carefully adjusted to the specific dataset in order to achieve the desired results.</h3>
