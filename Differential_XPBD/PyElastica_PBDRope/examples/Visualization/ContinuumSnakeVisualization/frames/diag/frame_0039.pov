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
    ,<0.14391546243359712,-0.0012146822745859628,0.654837208399314>,0.05
    ,<0.0995918492279906,-0.0011947159332594585,0.677976571043785>,0.05
    ,<0.054767969008129226,-0.0011792196688060325,0.7001411624042251>,0.05
    ,<0.010151512659240549,-0.0011611417498192428,0.7227292099173322>,0.05
    ,<-0.03283706604427168,-0.0011234570555723278,0.7482852264570694>,0.05
    ,<-0.07208657638486145,-0.001044979855617711,0.7792787052968565>,0.05
    ,<-0.10465382882389596,-0.0009089496595747181,0.8172283120470751>,0.05
    ,<-0.1267722207924884,-0.0007133361643398912,0.8620715484260415>,0.05
    ,<-0.13427127896606095,-0.00048197803796721866,0.9114980990701136>,0.05
    ,<-0.12429776158248534,-0.00025839735549592567,0.9604786234814284>,0.05
    ,<-0.09786834289978294,-8.643737320766738e-05,1.0029039547882246>,0.05
    ,<-0.05998233556451141,4.987826050842996e-06,1.035514937494714>,0.05
    ,<-0.016661554020700183,1.073847168274694e-05,1.0604696667505016>,0.05
    ,<0.027358906751288863,-5.436374427536694e-05,1.084176758512317>,0.05
    ,<0.06762976329300152,-0.00016623358534474474,1.1138085180023194>,0.05
    ,<0.09835593270478547,-0.00030170609940336174,1.1532457855162217>,0.05
    ,<0.11414386894241704,-0.0004459281311275569,1.2006764174328541>,0.05
    ,<0.11375597321493285,-0.0005954654492316195,1.2506613174548755>,0.05
    ,<0.10079541731801235,-0.0007553490851777577,1.2989381368177058>,0.05
    ,<0.08085604354959078,-0.000926898876193677,1.344778960246859>,0.05
    ,<0.05865187607284018,-0.0011036663881506383,1.3895709086972534>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
