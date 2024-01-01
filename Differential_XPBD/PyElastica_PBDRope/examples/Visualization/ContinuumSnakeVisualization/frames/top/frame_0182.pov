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
    ,<-0.16348586595346917,-0.0016554467550897868,4.494307509759528>,0.05
    ,<-0.20013324069190935,-0.0014166549841065983,4.528308860042082>,0.05
    ,<-0.23328306614391092,-0.0011650868556869673,4.56573081222067>,0.05
    ,<-0.25882277552558525,-0.0008949455446452094,4.608707534235243>,0.05
    ,<-0.27278329383268235,-0.0006180357317403802,4.656709890632238>,0.05
    ,<-0.2725564133576517,-0.00036838585599025285,4.706698273566375>,0.05
    ,<-0.25780902850586185,-0.00018546153435158738,4.754463561559322>,0.05
    ,<-0.23049305453690674,-9.308841969205749e-05,4.7963356989099>,0.05
    ,<-0.19410358667453162,-9.480987853403707e-05,4.830626652150892>,0.05
    ,<-0.15271913177799548,-0.00017488350334706162,4.858697633945912>,0.05
    ,<-0.11021270774516452,-0.00030397509953480184,4.885043918643175>,0.05
    ,<-0.0707583652397795,-0.0004471119396397061,4.915768917786962>,0.05
    ,<-0.0404012142887872,-0.0005720119699659357,4.955497166598614>,0.05
    ,<-0.026536623498279656,-0.0006463589970441301,5.00352725099726>,0.05
    ,<-0.03340532081172955,-0.0006473783440048608,5.053037598724589>,0.05
    ,<-0.05833828948197487,-0.0005800212736732131,5.096356526757271>,0.05
    ,<-0.09433243284142451,-0.000470996285508462,5.131036751501768>,0.05
    ,<-0.13500686509666143,-0.000348327550499834,5.160093372921382>,0.05
    ,<-0.1762830483077323,-0.00022635007809511704,5.188296350064779>,0.05
    ,<-0.21588090037330557,-0.0001078271371430495,5.218817082576358>,0.05
    ,<-0.2534222866995213,8.413762987002816e-06,5.25183908564397>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }