#include "../default.inc"

camera{
    location <15.0,10.5,-15.0>
    angle 30
    look_at <4.0,2.7,2.0>
    sky <0,1,0>
    right x*image_width/image_height
}
light_source{
    <15.0,10.5,-15.0>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <1500,2500,-1000>
    color White
}

sphere_sweep {
    linear_spline 21
    ,<-0.13713328009447345,-0.0013852876936430144,5.034893375142885>,0.05
    ,<-0.18007408472264197,-0.0012935548549785464,5.060495375253708>,0.05
    ,<-0.22097528351972928,-0.0011853572717823036,5.089244405324954>,0.05
    ,<-0.2567186539577843,-0.0010454863241144826,5.124194665926645>,0.05
    ,<-0.2833614743384906,-0.0008720569711441206,5.166487542724876>,0.05
    ,<-0.2970684823773453,-0.0006856315015466052,5.214549464850551>,0.05
    ,<-0.29544611869383247,-0.0005183801894137883,5.264498642813199>,0.05
    ,<-0.2784520947910921,-0.00039649660654897596,5.311498809679107>,0.05
    ,<-0.24847689058529684,-0.0003346550233152872,5.351499850639239>,0.05
    ,<-0.20973913259643193,-0.00033297674437058255,5.383107399925492>,0.05
    ,<-0.16685514529267723,-0.0003780668996071997,5.408826259500836>,0.05
    ,<-0.12389546942265427,-0.00044864658520075464,5.434420766048971>,0.05
    ,<-0.08550429203655831,-0.0005219980082169183,5.466451261426669>,0.05
    ,<-0.05839723579987353,-0.0005710393569375941,5.508449832631384>,0.05
    ,<-0.04945044644200883,-0.0005715469838233044,5.557618255115695>,0.05
    ,<-0.06061702871021629,-0.0005198259931831643,5.60632542835283>,0.05
    ,<-0.08738485208662272,-0.00043402123955037664,5.648525139029496>,0.05
    ,<-0.12286506886738402,-0.00034182762744121515,5.683726701445724>,0.05
    ,<-0.1615854630411632,-0.0002608596283169389,5.715342041271418>,0.05
    ,<-0.2002611225876514,-0.0001926632747678747,5.7470216207547>,0.05
    ,<-0.23769520256055449,-0.00013175342687238387,5.7801651558137905>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
