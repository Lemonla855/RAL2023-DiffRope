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
    ,<-0.03580670154546189,-0.00014161552859744695,0.008718670705289094>,0.05
    ,<-0.028956174596062534,-0.00010341805114953021,0.05824605665190982>,0.05
    ,<-0.020716563488694252,-6.67776548655183e-05,0.10756446873836974>,0.05
    ,<-0.010814184212133192,-3.49913316263344e-05,0.15657872857340863>,0.05
    ,<0.00032396505513667344,-1.1335164149107287e-05,0.20533013049187676>,0.05
    ,<0.01191159539909321,3.5302299248914457e-06,0.25397988008738015>,0.05
    ,<0.0230209709438174,1.1968143157511596e-05,0.3027439794028863>,0.05
    ,<0.03259934134660615,1.854556537578632e-05,0.35183394614915503>,0.05
    ,<0.039376144078316036,2.8338314332482815e-05,0.4013896314863746>,0.05
    ,<0.041958208410693594,4.43367485087479e-05,0.45133989320680684>,0.05
    ,<0.039306734889409535,6.460073158684901e-05,0.5012853675380932>,0.05
    ,<0.031258212221780005,8.09217672425995e-05,0.5506480489398371>,0.05
    ,<0.018870596828147408,8.463908475060565e-05,0.5991045822782404>,0.05
    ,<0.004328854262992811,7.30419318182357e-05,0.6469609079783065>,0.05
    ,<-0.009569916951956981,4.940931562153738e-05,0.6950111257749687>,0.05
    ,<-0.0200883727780349,2.0701476087819552e-05,0.7439126154709477>,0.05
    ,<-0.025262805703202752,-6.006093041262578e-06,0.7936604807612709>,0.05
    ,<-0.02444500642518571,-2.670073932316037e-05,0.8436657140054178>,0.05
    ,<-0.018397166655257292,-4.1738223884419764e-05,0.8933057028720104>,0.05
    ,<-0.008841970792810235,-5.4287322540701e-05,0.9423872848546132>,0.05
    ,<0.002233110582568842,-6.644528463723508e-05,0.99114515071147>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }