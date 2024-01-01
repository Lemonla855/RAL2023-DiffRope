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
    ,<-0.11655294152665704,-0.003136340668242379,0.20129485924426846>,0.05
    ,<-0.11912243596630166,-0.0025581186805931993,0.25119897084028914>,0.05
    ,<-0.1159373403217808,-0.0019856015974127244,0.30107168843204746>,0.05
    ,<-0.10367474326825796,-0.00143676121345592,0.3495241338661557>,0.05
    ,<-0.08170531791089934,-0.000944594552585524,0.39442481459763973>,0.05
    ,<-0.05144935762046617,-0.0005469805973771524,0.43422648210168063>,0.05
    ,<-0.015304070147272326,-0.0002685876770480531,0.46878137115164126>,0.05
    ,<0.024151407818644773,-0.00011114081709244333,0.4995145640298651>,0.05
    ,<0.06426392104865923,-5.491500408571154e-05,0.5293908457655419>,0.05
    ,<0.10168414189312125,-6.498427756234674e-05,0.5625729834599905>,0.05
    ,<0.13141223070414065,-0.00010124302125223086,0.6027803619409933>,0.05
    ,<0.14680683378472642,-0.00013198904558591037,0.650340401421501>,0.05
    ,<0.14250747517931156,-0.0001495072498214533,0.7001300056217088>,0.05
    ,<0.1187752028059975,-0.00017214554686648025,0.7441025997610514>,0.05
    ,<0.08170240398223455,-0.00022332586761561983,0.777611882915328>,0.05
    ,<0.03846478759000583,-0.0003129267997380484,0.8026858884746366>,0.05
    ,<-0.006057951967321619,-0.0004356307637153059,0.8254145937051712>,0.05
    ,<-0.04858766989615962,-0.0005767947052459384,0.8516869391918636>,0.05
    ,<-0.08639783053504481,-0.0007209403093087892,0.884387078944846>,0.05
    ,<-0.11832323985010207,-0.0008592029076367982,0.9228529907588296>,0.05
    ,<-0.14610741780864067,-0.0009918619096907322,0.9644106085910401>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
