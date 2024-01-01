#include "../default.inc"

camera{
    location <0,15,3>
    angle 30
    look_at <0.0,0,3>
    sky <-1,0,0>
    right x*image_width/image_height
}
light_source{
    <0.0,8.0,5.0>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <1500,2500,-1000>
    color White
}

sphere_sweep {
    linear_spline 21
    ,<-0.2977433768470847,-0.0007850979668015282,4.70727679337219>,0.05
    ,<-0.2564593564263235,-0.0008179917949147325,4.735479531590041>,0.05
    ,<-0.21426611746161145,-0.000857798305644114,4.762313048055759>,0.05
    ,<-0.1718037376670306,-0.0009003876086765485,4.788729374758202>,0.05
    ,<-0.13052647152134667,-0.0009306828680014882,4.816968222042656>,0.05
    ,<-0.0924875646350848,-0.0009265838451308899,4.849437777043315>,0.05
    ,<-0.06045857104121901,-0.0008662481805587886,4.887842680461519>,0.05
    ,<-0.03804882199582647,-0.0007376188299642982,4.932538479722786>,0.05
    ,<-0.029503683767547785,-0.0005509431950590257,4.981789815445283>,0.05
    ,<-0.03820354657387255,-0.00034219280516572484,5.0310044212600475>,0.05
    ,<-0.06391191989087427,-0.00015891751812863606,5.073860829916822>,0.05
    ,<-0.10202342792599199,-3.944078830001621e-05,5.106198970611198>,0.05
    ,<-0.14638027809759227,1.4429415558370162e-06,5.129257357094143>,0.05
    ,<-0.19226892205650212,-2.751348787084399e-05,5.149107926169852>,0.05
    ,<-0.23595909228404596,-0.00010421789638728896,5.173414340615405>,0.05
    ,<-0.2725137376282144,-0.00020464924865801218,5.207511980462025>,0.05
    ,<-0.2963505113513875,-0.0003117565541277725,5.251442712514634>,0.05
    ,<-0.3048570731086416,-0.00042025218179609145,5.300690476499955>,0.05
    ,<-0.30029652593545864,-0.0005349666145437488,5.350459696595024>,0.05
    ,<-0.28785188856523103,-0.0006582863872071113,5.3988684221080705>,0.05
    ,<-0.27256924808147087,-0.0007851294590210673,5.446462664137817>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
