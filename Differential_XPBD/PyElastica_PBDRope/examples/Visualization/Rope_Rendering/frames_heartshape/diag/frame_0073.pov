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
    ,<0.0,0.0,-30.0>,0.5559788942337036
    ,<-0.5845980048179626,1.144923210144043,-28.465396881103516>,0.5414279103279114
    ,<-1.257177472114563,2.2245843410491943,-26.919761657714844>,0.5476499795913696
    ,<-2.0828797817230225,3.189089059829712,-25.372583389282227>,0.5449594259262085
    ,<-3.095470428466797,3.962153196334839,-23.829504013061523>,0.5461745262145996
    ,<-4.482840061187744,4.388348579406738,-22.452762603759766>,0.5456952452659607
    ,<-5.823475360870361,4.639492034912109,-20.99008560180664>,0.5459461808204651
    ,<-7.4565300941467285,5.140805721282959,-19.950838088989258>,0.5458745360374451
    ,<-8.847114562988281,4.556747913360596,-18.638456344604492>,0.5458970665931702
    ,<-10.748543739318848,4.7322893142700195,-18.044200897216797>,0.5458411574363708
    ,<-12.192928314208984,4.569315433502197,-16.670063018798828>,0.5457906126976013
    ,<-13.982662200927734,5.009467124938965,-15.890710830688477>,0.5457520484924316
    ,<-15.515585899353027,5.596153736114502,-14.747133255004883>,0.5458252429962158
    ,<-17.001182556152344,6.2052903175354,-13.562028884887695>,0.5462932586669922
    ,<-18.489028930664062,7.1247334480285645,-12.630921363830566>,0.5481677055358887
    ,<-19.90285873413086,7.428484916687012,-11.315240859985352>,0.5457759499549866
    ,<-18.773605346679688,5.775178909301758,-10.679948806762695>,0.5318336486816406
    ,<-16.789527893066406,5.735843658447266,-9.894875526428223>,0.5259935259819031
    ,<-17.262866973876953,4.918316841125488,-7.555997848510742>,0.5044094324111938
    ,<-16.671091079711914,4.3321213722229,-4.650930404663086>,0.4829704463481903
    ,<-15.380778312683105,3.8496549129486084,-1.5005438327789307>,0.4745379090309143
    ,<-13.75329875946045,3.4139902591705322,1.7601364850997925>,0.47570499777793884
    ,<-12.028827667236328,2.988982915878296,5.048897743225098>,0.47476792335510254
    ,<-10.302362442016602,2.566136121749878,8.33045482635498>,0.47573232650756836
    ,<-8.586263656616211,2.1442935466766357,11.602364540100098>,0.47564247250556946
    ,<-6.872738838195801,1.7212989330291748,14.873079299926758>,0.4754025936126709
    ,<-5.157127380371094,1.2952420711517334,18.148235321044922>,0.4755830764770508
    ,<-3.438983917236328,0.8656443357467651,21.4287166595459>,0.47392016649246216
    ,<-1.7191879749298096,0.4331657886505127,24.713024139404297>,0.4795144200325012
    ,<0.0,0.0,28.0>,0.45802491903305054
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
