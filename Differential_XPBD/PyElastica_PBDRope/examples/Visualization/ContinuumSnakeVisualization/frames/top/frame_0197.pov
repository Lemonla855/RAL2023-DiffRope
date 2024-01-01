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
    ,<0.00430504210503775,-0.0013675694879926189,4.959454066489425>,0.05
    ,<-0.03374326362162202,-0.0012286978887295478,4.99187481428912>,0.05
    ,<-0.07380032652611587,-0.001102405784576181,5.021789587608426>,0.05
    ,<-0.1158009672032266,-0.0009964072587343267,5.048922779308673>,0.05
    ,<-0.1586516842813354,-0.0009101257692578304,5.074707140213444>,0.05
    ,<-0.20086511671305668,-0.0008319330110095805,5.10153007068598>,0.05
    ,<-0.24055417150442276,-0.0007433325499021111,5.131962990904489>,0.05
    ,<-0.27499986840764934,-0.0006252011983125704,5.168216048159023>,0.05
    ,<-0.3001112263831682,-0.00046677539902478435,5.211449228961933>,0.05
    ,<-0.3106915357632237,-0.0002786145961850739,5.260298574556634>,0.05
    ,<-0.30289057472624464,-9.712014140710944e-05,5.309656487503093>,0.05
    ,<-0.2774640963716339,3.372549780656634e-05,5.352673723691879>,0.05
    ,<-0.23989645881384153,8.642062063372967e-05,5.385636498514949>,0.05
    ,<-0.19672624576393538,5.877447207369104e-05,5.410841573346859>,0.05
    ,<-0.15289209013475208,-3.196241004992764e-05,5.4348833092918545>,0.05
    ,<-0.11271768583140074,-0.00016067462244778082,5.464633958554509>,0.05
    ,<-0.08126022409075913,-0.0003040486648821587,5.503476352694039>,0.05
    ,<-0.06260095559309267,-0.0004473621958581275,5.5498394090831225>,0.05
    ,<-0.056960093914693595,-0.0005874965029472326,5.599495056128075>,0.05
    ,<-0.06060385178495499,-0.000727898546636582,5.649339918332525>,0.05
    ,<-0.0683382480443397,-0.0008695152399052153,5.698720429364835>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
