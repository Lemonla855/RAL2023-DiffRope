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
    ,<0.0,0.0,-30.0>,0.5561634302139282
    ,<-0.2204088568687439,1.4491084814071655,-28.62740135192871>,0.5414260625839233
    ,<-0.5617930889129639,2.6990251541137695,-27.092693328857422>,0.5476925373077393
    ,<-1.167142629623413,3.7416789531707764,-25.484949111938477>,0.5448108315467834
    ,<-2.337080955505371,5.218212127685547,-24.783981323242188>,0.5463540554046631
    ,<-1.8918343782424927,6.357767581939697,-23.198476791381836>,0.546108067035675
    ,<-3.0723164081573486,6.033199787139893,-21.616159439086914>,0.5465803742408752
    ,<-2.347771644592285,7.78887414932251,-21.00703239440918>,0.546661376953125
    ,<-3.440584421157837,7.945037364959717,-19.345369338989258>,0.5466927289962769
    ,<-4.3358869552612305,9.391066551208496,-18.304359436035156>,0.5465212464332581
    ,<-4.639719486236572,7.7220635414123535,-17.245885848999023>,0.5461133718490601
    ,<-5.6094207763671875,6.583475589752197,-15.907466888427734>,0.5453020334243774
    ,<-7.274685859680176,5.4253034591674805,-15.942459106445312>,0.5433221459388733
    ,<-9.046418190002441,4.790260314941406,-16.807100296020508>,0.541204571723938
    ,<-10.81937026977539,4.27625036239624,-17.867347717285156>,0.530829131603241
    ,<-12.570907592773438,3.760435104370117,-19.030832290649414>,0.5521467328071594
    ,<-10.924132347106934,3.905898094177246,-18.12989616394043>,0.5636227130889893
    ,<-10.3615140914917,2.2460219860076904,-17.06313705444336>,0.5064345002174377
    ,<-9.657275199890137,3.107264995574951,-14.173999786376953>,0.4666118025779724
    ,<-8.960600852966309,3.3264994621276855,-10.451025009155273>,0.4674525260925293
    ,<-8.079533576965332,3.188875436782837,-6.474172115325928>,0.46372976899147034
    ,<-7.144276142120361,2.8822150230407715,-2.506621837615967>,0.4680914282798767
    ,<-6.199281692504883,2.54604434967041,1.3861398696899414>,0.468761682510376
    ,<-5.273063659667969,2.2102932929992676,5.227138042449951>,0.4701708257198334
    ,<-4.394619464874268,2.000120162963867,9.044214248657227>,0.4706365168094635
    ,<-3.476433515548706,1.501237392425537,12.812439918518066>,0.47029024362564087
    ,<-2.5980796813964844,1.1243773698806763,16.61187171936035>,0.4706588387489319
    ,<-1.7289434671401978,0.7498796582221985,20.409753799438477>,0.46955201029777527
    ,<-0.8646422028541565,0.3762151300907135,24.20589256286621>,0.4741101861000061
    ,<0.0,0.0,28.0>,0.45734086632728577
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
