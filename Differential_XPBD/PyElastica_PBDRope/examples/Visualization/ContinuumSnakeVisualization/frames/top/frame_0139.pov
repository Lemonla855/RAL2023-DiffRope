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
    ,<0.02212900629378172,-0.0007094716471622618,3.3784516851991113>,0.05
    ,<-0.023249734594327384,-0.0008023399760674461,3.3994495775783777>,0.05
    ,<-0.06874295461581427,-0.0008950905854191583,3.4202092726414444>,0.05
    ,<-0.1134261716040712,-0.000974897020174756,3.442668192837356>,0.05
    ,<-0.15560465542359375,-0.0010198499750973741,3.469540461331332>,0.05
    ,<-0.19274805098866082,-0.001006326947174247,3.5030264852660653>,0.05
    ,<-0.22149965878893432,-0.0009187924329121433,3.543937461233356>,0.05
    ,<-0.23797407641030635,-0.0007618739348208216,3.5911395846145977>,0.05
    ,<-0.23859088971164572,-0.0005640868193286913,3.641121996650922>,0.05
    ,<-0.22195398320216073,-0.00036804841695286416,3.6882544608731416>,0.05
    ,<-0.19065540970008427,-0.0002147352354436295,3.72722774477907>,0.05
    ,<-0.15025709436184048,-0.00012820216825025718,3.7566755305509933>,0.05
    ,<-0.10627977394445887,-0.00010979764209159546,3.7804624340025175>,0.05
    ,<-0.06308271823894553,-0.0001437972239474399,3.8056415858026766>,0.05
    ,<-0.02561918520002196,-0.00020755846926123555,3.8387485735731945>,0.05
    ,<-0.00025136289220056016,-0.0002811550786799819,3.881823043973211>,0.05
    ,<0.008330659126648463,-0.0003550189632114694,3.9310647117200483>,0.05
    ,<0.0005931213830945529,-0.00043272509693451956,3.9804438251942718>,0.05
    ,<-0.018618082237051026,-0.0005222079286511159,4.026588350500996>,0.05
    ,<-0.043554293695320725,-0.0006237318307385123,4.0699134652987325>,0.05
    ,<-0.07002511468527,-0.0007306711772145386,4.1123231604842685>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
