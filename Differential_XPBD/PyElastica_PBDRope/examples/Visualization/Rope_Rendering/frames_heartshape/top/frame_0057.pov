#include "../default.inc"

camera{
    location <0,200,3>
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
    linear_spline 30
    ,<0.0,0.0,-30.0>,0.5559284687042236
    ,<-0.3096909821033478,1.418544054031372,-28.59803581237793>,0.540888249874115
    ,<-0.7273135781288147,2.6648664474487305,-27.06729507446289>,0.5474234223365784
    ,<-1.2015479803085327,3.9133527278900146,-25.55705451965332>,0.5447661876678467
    ,<-0.9040887951850891,4.728041172027588,-23.74595832824707>,0.5471301078796387
    ,<-2.645334243774414,5.533101558685303,-23.244525909423828>,0.5476732850074768
    ,<-2.1919054985046387,6.698875904083252,-21.708972930908203>,0.5472405552864075
    ,<-3.6066935062408447,7.14585018157959,-20.376392364501953>,0.5453744530677795
    ,<-5.186035633087158,7.236733436584473,-19.117694854736328>,0.5445654392242432
    ,<-5.070319652557373,8.400224685668945,-17.470535278320312>,0.5468438267707825
    ,<-5.897943019866943,7.334568977355957,-18.93363380432129>,0.5460338592529297
    ,<-7.13958740234375,5.831837177276611,-18.385066986083984>,0.5440968871116638
    ,<-8.849259376525879,4.9450178146362305,-17.738901138305664>,0.5445937514305115
    ,<-10.692404747009277,4.142229080200195,-18.02863311767578>,0.5445438027381897
    ,<-12.349342346191406,3.1653518676757812,-18.675979614257812>,0.544930100440979
    ,<-14.21859359741211,2.535968065261841,-19.15565299987793>,0.5440858006477356
    ,<-15.731823921203613,3.770890951156616,-18.82887840270996>,0.5550147294998169
    ,<-13.8482084274292,4.1552557945251465,-17.785572052001953>,0.5035188794136047
    ,<-13.392175674438477,3.9127018451690674,-14.626110076904297>,0.4618319272994995
    ,<-12.521489143371582,3.545788049697876,-10.879912376403809>,0.46807384490966797
    ,<-11.341784477233887,3.1211133003234863,-6.9581074714660645>,0.46337994933128357
    ,<-10.040143966674805,2.7242603302001953,-3.0609524250030518>,0.46668481826782227
    ,<-8.739208221435547,2.364722728729248,0.7968869805335999>,0.4663412570953369
    ,<-7.466526508331299,2.018832206726074,4.649477481842041>,0.4666402339935303
    ,<-6.29344367980957,1.7711973190307617,8.536091804504395>,0.46629947423934937
    ,<-4.9671454429626465,1.3463667631149292,12.37244987487793>,0.4655947983264923
    ,<-3.7235662937164307,1.0054502487182617,16.268905639648438>,0.4655154347419739
    ,<-2.4828341007232666,0.6680320501327515,20.175235748291016>,0.46459904313087463
    ,<-1.2424689531326294,0.33434250950813293,24.0869197845459>,0.4675113558769226
    ,<0.0,0.0,28.0>,0.45642784237861633
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
