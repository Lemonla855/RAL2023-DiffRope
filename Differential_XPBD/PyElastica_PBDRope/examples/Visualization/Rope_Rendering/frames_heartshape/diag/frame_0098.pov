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
    ,<0.0,0.0,-30.0>,0.5464946627616882
    ,<-0.7488387823104858,0.839144766330719,-28.04286766052246>,0.5241987109184265
    ,<-1.497269630432129,1.6774849891662598,-26.08531379699707>,0.5339645147323608
    ,<-2.244894504547119,2.5142548084259033,-24.12691879272461>,0.5295525193214417
    ,<-2.9913463592529297,3.348778486251831,-22.167255401611328>,0.5315403938293457
    ,<-3.7363195419311523,4.180524826049805,-20.205833435058594>,0.5306435823440552
    ,<-4.479618072509766,5.009161472320557,-18.24205780029297>,0.5310110449790955
    ,<-5.221194744110107,5.834574222564697,-16.27525520324707>,0.5307603478431702
    ,<-5.961135387420654,6.656774520874023,-14.304957389831543>,0.5307441353797913
    ,<-6.699484825134277,7.475607872009277,-12.331548690795898>,0.5306597948074341
    ,<-7.435789585113525,8.290221214294434,-10.357240676879883>,0.5308330059051514
    ,<-8.168052673339844,9.098247528076172,-8.386992454528809>,0.5314429402351379
    ,<-8.893768310546875,9.898124694824219,-6.421003818511963>,0.5322405695915222
    ,<-9.621560096740723,10.700528144836426,-4.4305219650268555>,0.5305187702178955
    ,<-10.41292667388916,11.550893783569336,-2.364447593688965>,0.5218663215637207
    ,<-11.343255996704102,12.541112899780273,-0.22820965945720673>,0.5096785426139832
    ,<-12.526790618896484,13.482734680175781,1.8983485698699951>,0.5042740106582642
    ,<-13.071603775024414,13.999943733215332,4.274764537811279>,0.5427199602127075
    ,<-11.751067161560059,13.14706802368164,3.2776272296905518>,0.5504942536354065
    ,<-10.454377174377441,11.480690002441406,4.168686389923096>,0.5117266178131104
    ,<-9.451693534851074,10.238786697387695,6.237903594970703>,0.5014479160308838
    ,<-8.425151824951172,9.081332206726074,8.618279457092285>,0.5021792650222778
    ,<-7.355472564697266,7.926790237426758,11.075760841369629>,0.5012651085853577
    ,<-6.285468578338623,6.778885364532471,13.525123596191406>,0.5029993653297424
    ,<-5.229976654052734,5.643091201782227,15.952506065368652>,0.5036346912384033
    ,<-4.183650970458984,4.514910697937012,18.36699867248535>,0.5037475824356079
    ,<-3.139725685119629,3.3886539936065674,20.77682876586914>,0.5043225884437561
    ,<-2.0949745178222656,2.261190414428711,23.18531036376953>,0.5017510056495667
    ,<-1.0485599040985107,1.1315760612487793,25.59317970275879>,0.5122501254081726
    ,<0.0,0.0,28.0>,0.4655168652534485
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
