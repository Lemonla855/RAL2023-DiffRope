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
    ,<0.0,0.0,-30.0>,0.5556475520133972
    ,<-0.74115389585495,1.0371339321136475,-28.448286056518555>,0.5408743023872375
    ,<-1.558617115020752,2.017550230026245,-26.898025512695312>,0.5471853017807007
    ,<-2.5170085430145264,2.8817505836486816,-25.359344482421875>,0.5444211959838867
    ,<-3.6540651321411133,3.5796499252319336,-23.858234405517578>,0.5456109046936035
    ,<-4.960571765899658,4.093752861022949,-22.42197608947754>,0.5450708866119385
    ,<-6.386809349060059,4.454826354980469,-21.0542049407959>,0.5452792644500732
    ,<-7.984149932861328,4.687188148498535,-19.857898712158203>,0.5451610088348389
    ,<-9.558506965637207,4.917183876037598,-18.630428314208984>,0.5451764464378357
    ,<-11.2272367477417,5.1452436447143555,-17.533184051513672>,0.5451096892356873
    ,<-12.787992477416992,5.536679267883301,-16.3269100189209>,0.5450434684753418
    ,<-14.359556198120117,6.088166236877441,-15.197568893432617>,0.544942319393158
    ,<-15.828717231750488,6.840891361236572,-14.044360160827637>,0.544856071472168
    ,<-17.185686111450195,7.764442443847656,-12.876614570617676>,0.5449525117874146
    ,<-18.185577392578125,9.074959754943848,-11.719761848449707>,0.544556200504303
    ,<-19.700176239013672,8.702969551086426,-10.498045921325684>,0.5555660724639893
    ,<-19.10679054260254,6.810924053192139,-10.374274253845215>,0.5364814400672913
    ,<-17.27378273010254,7.106067180633545,-9.340474128723145>,0.525112509727478
    ,<-17.60031509399414,6.2534613609313965,-7.0628767013549805>,0.5103418827056885
    ,<-16.75075340270996,5.580081939697266,-4.295132160186768>,0.4818854033946991
    ,<-15.356839179992676,5.009541988372803,-1.2140079736709595>,0.47405821084976196
    ,<-13.688313484191895,4.465850353240967,2.024080753326416>,0.475202351808548
    ,<-11.946391105651855,3.9165689945220947,5.296664714813232>,0.47480371594429016
    ,<-10.217140197753906,3.3570683002471924,8.551567077636719>,0.4761030972003937
    ,<-8.507980346679688,2.793895721435547,11.788028717041016>,0.4761924147605896
    ,<-6.806771278381348,2.2314765453338623,15.020301818847656>,0.475974440574646
    ,<-5.106259346008301,1.6710753440856934,18.257362365722656>,0.47618257999420166
    ,<-3.404737710952759,1.112696886062622,21.500919342041016>,0.47447845339775085
    ,<-1.702459692955017,0.55588299036026,24.749256134033203>,0.48023128509521484
    ,<0.0,0.0,28.0>,0.45810365676879883
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
