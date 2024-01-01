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
    ,<-0.30591811049424406,-0.002081149447493795,4.577372358947141>,0.05
    ,<-0.2973163261745553,-0.0016572213093218318,4.626606350164275>,0.05
    ,<-0.2833760584030836,-0.0012435785476269576,4.6746075907823545>,0.05
    ,<-0.26174058478043416,-0.0008595130376695525,4.719673494504198>,0.05
    ,<-0.23266065905127725,-0.0005339446949956625,4.760343176178329>,0.05
    ,<-0.19791320763580209,-0.00029098480973042685,4.796301860842582>,0.05
    ,<-0.15975560242990602,-0.00013632381819599424,4.828629771760953>,0.05
    ,<-0.12048174066362881,-5.726178909646393e-05,4.859598715693244>,0.05
    ,<-0.08278184702342303,-2.7401139930394868e-05,4.89246612499283>,0.05
    ,<-0.050640297532360316,-1.4506489823494299e-05,4.930780349376554>,0.05
    ,<-0.02984061049834522,8.51630776625487e-06,4.976249145294301>,0.05
    ,<-0.026484018001535808,4.980122568080153e-05,5.02612307736292>,0.05
    ,<-0.043038270084535445,9.049946535901387e-05,5.073277124662895>,0.05
    ,<-0.07575532876141997,0.00010252059576432125,5.111052900279366>,0.05
    ,<-0.11747417810932739,6.847956273958421e-05,5.1385759189128315>,0.05
    ,<-0.16224462237585288,-1.4006363914639452e-05,5.160806597370607>,0.05
    ,<-0.20621817110682017,-0.00013243949108867676,5.184580533887031>,0.05
    ,<-0.24602897394075166,-0.0002669181045192803,5.214811954219792>,0.05
    ,<-0.27880028858836814,-0.00039959372562505387,5.252558326891871>,0.05
    ,<-0.30406313454232337,-0.0005217588127359616,5.295692712775821>,0.05
    ,<-0.32463509259441053,-0.0006351675936659228,5.341253506581509>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
