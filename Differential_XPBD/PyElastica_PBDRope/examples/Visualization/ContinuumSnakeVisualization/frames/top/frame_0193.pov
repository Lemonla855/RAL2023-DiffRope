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
    ,<-0.07869625276534796,-0.0021493616198161266,4.784804335468968>,0.05
    ,<-0.058089403856931045,-0.0017559839214884192,4.83033199141944>,0.05
    ,<-0.04276495507418312,-0.0013562306614687547,4.877900402360979>,0.05
    ,<-0.03717365377294012,-0.0009555385900019136,4.927564142222226>,0.05
    ,<-0.043924642384932315,-0.0005782788311239523,4.977086441844436>,0.05
    ,<-0.06323138789122293,-0.0002665091371416069,5.023191592828308>,0.05
    ,<-0.09322971397532602,-5.969086862026562e-05,5.063182592943523>,0.05
    ,<-0.13089549485274948,2.561506028329371e-05,5.09606636098718>,0.05
    ,<-0.17295928718450987,-2.408760418889017e-06,5.123111502256941>,0.05
    ,<-0.2163069627616208,-0.00011521634647257909,5.148053142832411>,0.05
    ,<-0.25754050154991853,-0.00027195977557883174,5.1763466340242275>,0.05
    ,<-0.2915951123594472,-0.0004287825932013693,5.212951985124708>,0.05
    ,<-0.3113825512504056,-0.0005495066822202544,5.258851005900602>,0.05
    ,<-0.31115153633677933,-0.0006081898435519639,5.308823838737483>,0.05
    ,<-0.2912534586397969,-0.0005985722770870616,5.354662278132211>,0.05
    ,<-0.25790359971096066,-0.0005413091013813933,5.391881640010471>,0.05
    ,<-0.21822844598142374,-0.00046416044051538895,5.422280836136052>,0.05
    ,<-0.17728395023346671,-0.00038254386794083894,5.450957663248512>,0.05
    ,<-0.13837001542155764,-0.0002984575114369221,5.482339423715978>,0.05
    ,<-0.1033034062810402,-0.00020953464250644653,5.517970966077177>,0.05
    ,<-0.07150667863696576,-0.00011630618310013967,5.556551533879587>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
