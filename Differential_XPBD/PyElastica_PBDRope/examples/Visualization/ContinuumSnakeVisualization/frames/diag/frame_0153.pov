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
    ,<-0.0447531418305085,-0.0020295347214999027,3.694445080195711>,0.05
    ,<-0.02151845812347207,-0.0016613663851678417,3.7386901855394923>,0.05
    ,<-0.0033530808218476,-0.0012855861447372478,3.7852481671062703>,0.05
    ,<0.005238783303059173,-0.000905657236428047,3.8344813091317738>,0.05
    ,<0.0013569037687778198,-0.0005440383647293286,3.8843094596257925>,0.05
    ,<-0.015570255834584967,-0.00024199575803888262,3.9313382106450394>,0.05
    ,<-0.04391123274192437,-3.969807869325584e-05,3.9725172192793674>,0.05
    ,<-0.08067022868540101,4.4561748211410766e-05,4.006407988482157>,0.05
    ,<-0.12245758327864376,1.6885244931412024e-05,4.0338749542533625>,0.05
    ,<-0.1660441462013711,-9.601046854191564e-05,4.058395290075487>,0.05
    ,<-0.2081153288921777,-0.00025454162884327015,4.085427757270849>,0.05
    ,<-0.24390803241199258,-0.0004153167509965986,4.120336107038858>,0.05
    ,<-0.2664638142423537,-0.0005412840761396339,4.164939765316987>,0.05
    ,<-0.2694866214246514,-0.0006044394988267592,4.21482039726488>,0.05
    ,<-0.25235140268608225,-0.0005957500403736803,4.26175968532652>,0.05
    ,<-0.22074497376977223,-0.0005333351319509005,4.3004683497265646>,0.05
    ,<-0.18192205581119547,-0.00044613625355251256,4.331946411303879>,0.05
    ,<-0.1412254842924573,-0.0003529855463811835,4.360972922263692>,0.05
    ,<-0.10209988872858558,-0.000257820641216848,4.3920905252642735>,0.05
    ,<-0.06646105525125459,-0.00015902370322799404,4.427150433800002>,0.05
    ,<-0.03391076036703734,-5.688978152060812e-05,4.4650981857368395>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
