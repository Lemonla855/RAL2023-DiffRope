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
    ,<-0.014918723471410508,-0.0014073985412474536,0.9613742330493205>,0.05
    ,<0.023864103645102202,-0.001264066086239903,0.9929226270423128>,0.05
    ,<0.05986480827450989,-0.0011035393857331286,1.0276142919661528>,0.05
    ,<0.08935815320047345,-0.0009124135619123806,1.067982768394316>,0.05
    ,<0.10824196225829158,-0.0006927059502376943,1.1142701086149012>,0.05
    ,<0.11317689302927535,-0.0004696795173311653,1.1640123344682216>,0.05
    ,<0.1027981102824856,-0.00027905239511603083,1.212908155912518>,0.05
    ,<0.07821511772972244,-0.0001479246414328964,1.2564338293863744>,0.05
    ,<0.04261917266801324,-8.896755397477018e-05,1.2915382163351539>,0.05
    ,<0.0003152281425257721,-9.815743377600134e-05,1.3181932343651288>,0.05
    ,<-0.04464672213088199,-0.00015720707618400184,1.340079538502925>,0.05
    ,<-0.08879982195970831,-0.00024011555612648224,1.363553915851903>,0.05
    ,<-0.1275845126813036,-0.0003210674465154864,1.3951072968534568>,0.05
    ,<-0.15405507546171432,-0.00037184654877567367,1.4375142774815386>,0.05
    ,<-0.16161002343735933,-0.00036920525695540314,1.486923048238001>,0.05
    ,<-0.1492293140894375,-0.00031250623130769375,1.5353449806320778>,0.05
    ,<-0.1221343852324659,-0.00022235453018396668,1.5773446551807835>,0.05
    ,<-0.08727620957929252,-0.00012588066629195163,1.6131697365761628>,0.05
    ,<-0.04996321389107647,-3.8109501866304954e-05,1.6464384246622334>,0.05
    ,<-0.01336140413629297,4.1190872377375054e-05,1.6804946501378402>,0.05
    ,<0.021562725599750837,0.00011691276859971902,1.7162737502444805>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
