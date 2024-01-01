#include "../default.inc"

camera{
    location <40.0,100.5,-40.0>
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
    linear_spline 30
    ,<0.0,0.0,-30.0>,0.5551963448524475
    ,<-0.995995044708252,0.8495122194290161,-28.457077026367188>,0.5399187207221985
    ,<-2.034120559692383,1.6635493040084839,-26.92271614074707>,0.5464518666267395
    ,<-3.1495790481567383,2.4123332500457764,-25.409269332885742>,0.5435602068901062
    ,<-4.364899635314941,3.078214168548584,-23.93393898010254>,0.5447794795036316
    ,<-5.685477256774902,3.6582705974578857,-22.512964248657227>,0.5441834926605225
    ,<-7.160783290863037,4.155725479125977,-21.21718978881836>,0.5443657040596008
    ,<-8.392476081848145,4.832798480987549,-19.75718879699707>,0.544188380241394
    ,<-10.03891658782959,5.151919364929199,-18.6171932220459>,0.5441499948501587
    ,<-11.331278800964355,5.864462375640869,-17.224624633789062>,0.5440545082092285
    ,<-13.036416053771973,6.274138927459717,-16.202739715576172>,0.544066309928894
    ,<-14.29504108428955,7.1252121925354,-14.858607292175293>,0.5441514849662781
    ,<-15.751733779907227,7.906156539916992,-13.68262004852295>,0.5442767143249512
    ,<-17.176528930664062,8.766213417053223,-12.51414680480957>,0.5440011024475098
    ,<-18.54299545288086,9.789931297302246,-11.387113571166992>,0.5405105948448181
    ,<-20.064708709716797,10.116585731506348,-10.0780668258667>,0.552379846572876
    ,<-18.52387809753418,8.886825561523438,-10.224437713623047>,0.537428617477417
    ,<-16.918779373168945,9.482447624206543,-8.929854393005371>,0.5225946307182312
    ,<-17.08502197265625,7.985727787017822,-6.93598747253418>,0.5066438913345337
    ,<-16.355260848999023,6.961918354034424,-4.164087295532227>,0.48062050342559814
    ,<-15.021632194519043,6.159983158111572,-1.0439362525939941>,0.47271373867988586
    ,<-13.390427589416504,5.449387073516846,2.2237138748168945>,0.474653035402298
    ,<-11.678783416748047,4.757006645202637,5.507339954376221>,0.4747737944126129
    ,<-9.980401039123535,4.067171096801758,8.757630348205566>,0.4763910174369812
    ,<-8.305668830871582,3.3825325965881348,11.978042602539062>,0.47682249546051025
    ,<-6.642185211181641,2.703026056289673,15.185050964355469>,0.4768393337726593
    ,<-4.981584072113037,2.026329755783081,18.38886070251465>,0.47726932168006897
    ,<-3.321209669113159,1.3507007360458374,21.592575073242188>,0.4756161868572235
    ,<-1.6606968641281128,0.6753628253936768,24.796377182006836>,0.4817700982093811
    ,<0.0,0.0,28.0>,0.4583939015865326
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
