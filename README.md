# XPBD


## Coding part 
This is the coding for the paper https://arxiv.org/pdf/2202.09714.pdf. This coding includes all the source data and code, and every file can be run directly without any change.

The structure of the coding part:

```
├── Differential_XPBD       # Files for Inextensible Rope Parameter Identification
│   ├── script              # scrips files for Inextensible Rope Parameter Identification
│    ├── PBDRope.py         # position estimation of the control point
│    ├── para_estimation.py # parameter identification for Inextensible Rope
│    ├── result_analysis.py # result analysis for Inextensible Rope
│    └── Visulization.py    # simulation and result visulization
│   ├── PyElastica_PBDRope  # Visualization for simulation by PyElastica
│
└── DVRKROPE                # Files for Extensible Rope Parameter Identification and Key Points Estimation
│   ├── script              # scrips files for extensible Rope Parameter Identification
│    ├── DVRKRope.py        # position estimation of the control point
│    ├── para_estimation.py # parameter identification for extensible Rope
│    ├── analysis_result.py # result analysis for extensible Rope
│    └── Visulization.py    # simulation and result visulization
│ 
└── Baxter_0928             # Files for Inextensible Rope Shape Control
│   ├── script              # scrips files for Inextensible Rope Shape Control
│    ├── PBDRope.py         # shape control 
└──
```

## Solver part

In the paper, we have used three constraints to simulate the deformable rope, including Distance Constraint, bending constraint, Shear and Stretch Constraint. The coding for the solver part is designed specifically for the Rope-like Objects. Some modification is necessary if you want to use these for multi ropes objects, like muscles. 

 
 
 ### Distance solver
 
 There are two kinds of solvers for the distance constraint, including the XPBD solver and the Thomas solver. The main difference between these two kinds of solvers is that the XPBD solver is an iterative method, which means that it needs to iterate infinite times to ensure the distance between two particles is unchanged. But the Thomas solver can preserve the inextensible characteristics of rope-like with several iterations. <b>The Thomas solver is only applicable to inextensible rope, and the XPBD solver applies to both inextensible and extensible rope.</b>
 
 
 ### Parameter 
 
 To obtain the ideal real2sim result, some parameters need to be identified. In the experiment, all the parameters have been optimized by backpropagation. The file can be found in ```../Differential_XPBD/script/para_estimation.py```.
 
 
Considering that the experiment's files are so large, you only need to refer to ```XPBD.py```, which has been included as the most crucial part of this project. Most of the constraints are referred to from https://github.com/vcg-uvic/viper. The Cuda version of most constraints can be found at https://github.com/vcg-uvic/viper/blob/master/Viper/CudaSolver.cu, and C++ version can be found at https://github.com/InteractiveComputerGraphics/PositionBasedDynamics/blob/master/PositionBasedDynamics/XPBD.cpp.

 
 ### Bonus
 
 In the ```XPBD.py```, other constraints are not used for this paper. It includes two kinds of volume constraints, radius constraints, and shape matching constraints. For detailed information about this part, you can refer to the paper https://arxiv.org/pdf/1906.05260.pdf.
 
 <b> Most coding is written specifically for the rope-like object. If you need to use it for another type of object, some changes are necessary.</b>

If you have any questions about the coding part, I am happy to fix the potential bug.



 
 
