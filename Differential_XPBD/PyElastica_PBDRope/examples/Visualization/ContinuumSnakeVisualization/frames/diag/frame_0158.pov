#include "../default.inc"

camera{
    location <15.0,10.5,-15.0>
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
    linear_spline 21
    ,<0.028729120850465292,-0.0009435797430997088,3.9004276591318288>,0.05
    ,<-0.014116339594897621,-0.0009474869289329231,3.926192385399267>,0.05
    ,<-0.05790385810250358,-0.0009591597136913566,3.950333738877163>,0.05
    ,<-0.1021312492288272,-0.000975973332303081,3.973672945462441>,0.05
    ,<-0.14557183517254516,-0.0009845490235509226,3.9984545118998795>,0.05
    ,<-0.18645823470029177,-0.0009639715178113208,4.027256224576227>,0.05
    ,<-0.22229218842721188,-0.0008926119653274478,4.062137425241247>,0.05
    ,<-0.24961265563978205,-0.0007569513372981716,4.104011893367563>,0.05
    ,<-0.2640263762638046,-0.000563214765590983,4.151874862545282>,0.05
    ,<-0.2614776980230947,-0.00034531527507617615,4.2017849632544495>,0.05
    ,<-0.24113751054324223,-0.00015214678342715835,4.2474301719904>,0.05
    ,<-0.2068759538294014,-2.3359439133771248e-05,4.283816866802446>,0.05
    ,<-0.1649630710654816,2.498254692460501e-05,4.3110614159686>,0.05
    ,<-0.12078440062326647,7.505794518035066e-07,4.334469912060272>,0.05
    ,<-0.07865962314079805,-7.434516485362437e-05,4.361398544722807>,0.05
    ,<-0.043739144840742566,-0.00017611841317113965,4.397167471487515>,0.05
    ,<-0.02160005777627237,-0.0002867202957332023,4.441976209119389>,0.05
    ,<-0.014870310238922732,-0.0003996822225078294,4.491495569230419>,0.05
    ,<-0.02134645329718193,-0.0005189022979007205,4.541049081069843>,0.05
    ,<-0.03587721511916218,-0.0006463642322245252,4.588870404932225>,0.05
    ,<-0.053384354842594774,-0.0007769801467774593,4.635689643109476>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }