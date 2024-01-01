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
    ,<0.12231423133515713,-0.0008331412791890225,1.7168017586922704>,0.05
    ,<0.08131852033730191,-0.0008158882717871927,1.7454174892649232>,0.05
    ,<0.0389478091577552,-0.000808781581429782,1.7719650746746736>,0.05
    ,<-0.0044722683700241855,-0.0008139513461107588,1.796771639388767>,0.05
    ,<-0.04780270257591051,-0.0008231934895481515,1.8217445907474108>,0.05
    ,<-0.08945640195632532,-0.0008179909100169504,1.8494281858012356>,0.05
    ,<-0.12724837290781302,-0.000775588301155566,1.8821852055506374>,0.05
    ,<-0.15803038279221132,-0.0006773996331421708,1.921595119507986>,0.05
    ,<-0.17743774704837817,-0.0005198093914764617,1.9676719083478522>,0.05
    ,<-0.1807229990442766,-0.0003266181842759697,2.0175491822116345>,0.05
    ,<-0.16563127401267352,-0.00014134281108514618,2.0651946341271445>,0.05
    ,<-0.13482062972586892,-4.565443376464677e-06,2.1045480584386445>,0.05
    ,<-0.09444712170406558,6.0931200634132046e-05,2.1340221214453066>,0.05
    ,<-0.05044450566339301,5.554371894326013e-05,2.157755965740725>,0.05
    ,<-0.007257973756113225,-3.2035030254171453e-06,2.1829494455929024>,0.05
    ,<0.030438450104636958,-9.102361865586643e-05,2.2157887244475876>,0.05
    ,<0.05720703176737656,-0.00018671675096967622,2.2580055427509165>,0.05
    ,<0.06958325005900606,-0.0002797258702850508,2.3064321073261733>,0.05
    ,<0.06861458911691246,-0.0003720762410272998,2.3564042302691712>,0.05
    ,<0.05889521672160983,-0.0004669431692088496,2.405434639688108>,0.05
    ,<0.04565733704460456,-0.0005620243209103345,2.453638497856396>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
