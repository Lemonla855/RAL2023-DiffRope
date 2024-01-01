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
    ,<0.10019446712199771,-0.0013498457558243275,2.262800664853109>,0.05
    ,<0.058656623906356675,-0.0012714933888594191,2.2906251452401025>,0.05
    ,<0.01586233637275982,-0.0012032415811609658,2.316485625711397>,0.05
    ,<-0.027816347052668368,-0.001146879562516698,2.34083455111998>,0.05
    ,<-0.07122579804804188,-0.0010940010653282316,2.3656696159110457>,0.05
    ,<-0.11274462959207927,-0.0010267906800176814,2.393553297235376>,0.05
    ,<-0.1501164428475919,-0.0009243754072075949,2.426786008086885>,0.05
    ,<-0.18011123052722752,-0.0007712903378552296,2.4667950844536337>,0.05
    ,<-0.1983409805842936,-0.0005680800478188847,2.513347154950365>,0.05
    ,<-0.20020158984681796,-0.0003428758716371425,2.563295380431131>,0.05
    ,<-0.18377804573907125,-0.0001415620341030811,2.610497038729278>,0.05
    ,<-0.15205571118002287,-3.2486958160046993e-06,2.6491195957434956>,0.05
    ,<-0.11125397982730889,5.388247314055231e-05,2.6779995525649642>,0.05
    ,<-0.06717486358266381,3.5318579500565986e-05,2.7015922692228664>,0.05
    ,<-0.0242125835462826,-3.8092678296565915e-05,2.7271648794883827>,0.05
    ,<0.012849956866070305,-0.00014114565310796375,2.7607153162491884>,0.05
    ,<0.03851932897039795,-0.00025350407041004026,2.803606668871673>,0.05
    ,<0.04951137723397654,-0.0003659484329452178,2.85236411322433>,0.05
    ,<0.047157970508996776,-0.0004809879515510078,2.9022889583203506>,0.05
    ,<0.03620794176893183,-0.0006010607428040567,2.951058728650163>,0.05
    ,<0.0218692306124292,-0.0007225398319055954,2.998946397018072>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
