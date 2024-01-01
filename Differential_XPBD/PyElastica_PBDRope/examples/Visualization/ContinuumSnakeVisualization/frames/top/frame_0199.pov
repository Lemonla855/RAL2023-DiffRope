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
    ,<-0.052791668918962795,-0.0008013396681650223,5.012951846853644>,0.05
    ,<-0.09849590149215767,-0.0008938226385264163,5.0332310610039>,0.05
    ,<-0.14410756266983585,-0.0009837065306301858,5.053728168107375>,0.05
    ,<-0.18853451638974772,-0.0010552965885126052,5.076686760105507>,0.05
    ,<-0.22985326185429447,-0.0010852079573666737,5.104856258960524>,0.05
    ,<-0.26524438088837615,-0.0010511466361295472,5.1401788265221136>,0.05
    ,<-0.29114428939401654,-0.0009428042794603201,5.182939666332037>,0.05
    ,<-0.3037426755824653,-0.000773187905737999,5.231307509966105>,0.05
    ,<-0.3000138000260381,-0.0005750676241237535,5.281142533236531>,0.05
    ,<-0.2795336411807163,-0.00038818253194453637,5.326727539870912>,0.05
    ,<-0.24569759594747684,-0.00024813456573775184,5.363515185298654>,0.05
    ,<-0.2041506155953644,-0.0001736393397002792,5.391321670437205>,0.05
    ,<-0.16003596044089274,-0.0001620960914463741,5.414859291440475>,0.05
    ,<-0.11762100707790706,-0.00019527294074145335,5.441335969433842>,0.05
    ,<-0.0822567806443967,-0.0002494503878388034,5.4766706546481485>,0.05
    ,<-0.060488481834781546,-0.000305877179307934,5.521661729119011>,0.05
    ,<-0.056323539684163815,-0.0003589633795430349,5.571460807490999>,0.05
    ,<-0.06823481323576537,-0.00041773465519687,5.619992009634558>,0.05
    ,<-0.09080028156871603,-0.0004914579615535678,5.664584519515065>,0.05
    ,<-0.11832622267744008,-0.0005784639471921613,5.706306882630421>,0.05
    ,<-0.146978019777956,-0.0006714793567894397,5.747271206554059>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
