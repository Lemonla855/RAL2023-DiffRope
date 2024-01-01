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
    ,<-0.050194586216967964,-0.001839520788125101,2.048950982160592>,0.05
    ,<-0.01264752180992424,-0.0016439310641384434,2.0819640423887216>,0.05
    ,<0.021831052301202337,-0.001433788097565295,2.1181731542206466>,0.05
    ,<0.0493312576839087,-0.0011992348589512257,2.159929709603561>,0.05
    ,<0.06576164447323057,-0.0009464280090068085,2.207149387544583>,0.05
    ,<0.06803839532184845,-0.0007042129104034068,2.257090246267997>,0.05
    ,<0.055207455556562875,-0.0005096591224467102,2.3054071513265724>,0.05
    ,<0.028759722743679356,-0.00038802689236964435,2.347832046162519>,0.05
    ,<-0.0079278284455692,-0.00034755388468062814,2.3817999452685643>,0.05
    ,<-0.05062218325148008,-0.00037881285875847976,2.407829464195675>,0.05
    ,<-0.09545345213913163,-0.00045868921937863063,2.4299852354723654>,0.05
    ,<-0.13892929951362265,-0.0005574509692824806,2.4546953301366154>,0.05
    ,<-0.17614964650919224,-0.0006466329437716653,2.488085370901582>,0.05
    ,<-0.19992403487276547,-0.0006962125291557854,2.532067778424078>,0.05
    ,<-0.2041020777272399,-0.0006825241815326368,2.581883092801855>,0.05
    ,<-0.1886346281964185,-0.0006061023150859372,2.6294158491362953>,0.05
    ,<-0.15935664289614443,-0.0004890871481643338,2.669928986918393>,0.05
    ,<-0.12318378635566408,-0.00036021022462130194,2.704428688765221>,0.05
    ,<-0.08514273059207422,-0.00023698918443046174,2.7368626998444703>,0.05
    ,<-0.04816891292280835,-0.00012216878026954673,2.7705141905845014>,0.05
    ,<-0.013040495949076792,-1.2508724019213201e-05,2.806092567724618>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
