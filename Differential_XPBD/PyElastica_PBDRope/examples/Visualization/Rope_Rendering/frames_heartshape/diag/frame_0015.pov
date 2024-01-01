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
    b_spline 30
    ,<0.0,0.0,-7.5>,0.1995701938867569
    ,<-0.060316022485494614,0.015547975897789001,-7.128726005554199>,0.1958923488855362
    ,<-0.12094169110059738,0.031179727986454964,-6.755641937255859>,0.19608920812606812
    ,<-0.1809556931257248,0.04666360467672348,-6.386407375335693>,0.19608338177204132
    ,<-0.24184675514698029,0.062396496534347534,-6.011653423309326>,0.19609083235263824
    ,<-0.30117249488830566,0.07776537537574768,-5.645988941192627>,0.19613520801067352
    ,<-0.36181408166885376,0.09354764968156815,-5.2709479331970215>,0.1962267905473709
    ,<-0.4194253385066986,0.10864628851413727,-4.912779331207275>,0.19639188051223755
    ,<-0.47898799180984497,0.12441467493772507,-4.5403947830200195>,0.19664879143238068
    ,<-0.5340946912765503,0.13915936648845673,-4.195908546447754>,0.19697703421115875
    ,<-0.5946728587150574,0.15537972748279572,-3.8266677856445312>,0.19732660055160522
    ,<-0.6518844366073608,0.17030346393585205,-3.500962018966675>,0.19759979844093323
    ,<-0.7274839282035828,0.18870113790035248,-3.1234230995178223>,0.1975722461938858
    ,<-0.7967889308929443,0.20439760386943817,-2.812558650970459>,0.19713982939720154
    ,<-0.8908926844596863,0.22770698368549347,-2.390723705291748>,0.19586805999279022
    ,<-0.9496616125106812,0.24330374598503113,-2.0676989555358887>,0.1937510222196579
    ,<-0.9813496470451355,0.2496829777956009,-1.534185528755188>,0.19021695852279663
    ,<-0.980853796005249,0.25037601590156555,-1.1508471965789795>,0.1862691342830658
    ,<-0.9248805046081543,0.23755696415901184,-0.47547346353530884>,0.18265050649642944
    ,<-0.8765732049942017,0.22519899904727936,-0.022223055362701416>,0.1794210523366928
    ,<-0.783976674079895,0.20156070590019226,0.7581557035446167>,0.17730364203453064
    ,<-0.7175987958908081,0.18456946313381195,1.2864067554473877>,0.1751565784215927
    ,<-0.6121466159820557,0.1576104760169983,2.113675355911255>,0.17471948266029358
    ,<-0.5364708304405212,0.13819408416748047,2.7050998210906982>,0.1737806499004364
    ,<-0.4330644905567169,0.11150467395782471,3.518526315689087>,0.17442187666893005
    ,<-0.35311803221702576,0.09069337695837021,4.153871536254883>,0.174215167760849
    ,<-0.25707483291625977,0.06567548960447311,4.9226508140563965>,0.17511752247810364
    ,<-0.18342846632003784,0.04006827995181084,5.587676048278809>,0.17517858743667603
    ,<-0.08560295403003693,0.023181110620498657,6.309722900390625>,0.17496518790721893
    ,<0.0,0.0,7.0>,0.18410666286945343
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
