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
    ,<-0.01470406717114051,-0.0007217793858714324,3.3950718858547804>,0.05
    ,<-0.06066911429277319,-0.0008241959717783317,3.414755009279937>,0.05
    ,<-0.10599958130693077,-0.000915997216249907,3.435867386985997>,0.05
    ,<-0.14905000945594896,-0.0009753040472122733,3.4613110741371202>,0.05
    ,<-0.18715318948927315,-0.000979330587498426,3.493693218660351>,0.05
    ,<-0.21673177537403063,-0.0009171274765188832,3.534002915836096>,0.05
    ,<-0.23395515556125426,-0.0007976685763996796,3.580931417458622>,0.05
    ,<-0.2357940625227565,-0.0006454711260149486,3.6308806905511304>,0.05
    ,<-0.22117251330188964,-0.0004923685300712341,3.678675571507047>,0.05
    ,<-0.192054127264916,-0.0003697286268294157,3.719303818552555>,0.05
    ,<-0.1531742278056791,-0.000298025268720723,3.750731430531374>,0.05
    ,<-0.10980333164638999,-0.00027987221269125374,3.7756129612712566>,0.05
    ,<-0.06629382922269464,-0.00030186821562458536,3.800256471126374>,0.05
    ,<-0.02731032953755466,-0.0003411765292440028,3.831566129797325>,0.05
    ,<0.0006011712898329211,-0.0003751193897881651,3.8730408561572416>,0.05
    ,<0.011209132496658955,-0.000392396499162265,3.921886420204808>,0.05
    ,<0.0031980397027902407,-0.0003984778069395291,3.971220225383437>,0.05
    ,<-0.019117808723398487,-0.0004112028248453782,4.015942003316873>,0.05
    ,<-0.0494359462435447,-0.00044238940369790487,4.055683131007315>,0.05
    ,<-0.08267922575974436,-0.0004890818276377905,4.09301949106202>,0.05
    ,<-0.11604871014430061,-0.0005427166596141521,4.13024922012621>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
