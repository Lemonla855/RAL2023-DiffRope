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
    ,<0.10756134598881852,-0.003068571804570454,0.4474473716425438>,0.05
    ,<0.12408246705469367,-0.002601589914824987,0.4946178817407101>,0.05
    ,<0.13506233215586422,-0.0021303429870853027,0.5433793416060155>,0.05
    ,<0.13619383568275603,-0.0016620603948256925,0.5933512203153479>,0.05
    ,<0.12532618728256156,-0.0012218114083443492,0.6421430215095071>,0.05
    ,<0.1027395914123231,-0.0008503705571713722,0.6867403701645823>,0.05
    ,<0.07056538287297048,-0.0005845314753403407,0.7250090841895037>,0.05
    ,<0.03180109105916331,-0.0004383365383864434,0.7565953990357972>,0.05
    ,<-0.010479020520870951,-0.00040213572237534896,0.78330379506974>,0.05
    ,<-0.053283754242001255,-0.0004472075267744157,0.8091675187788067>,0.05
    ,<-0.09301864013430555,-0.000533635926442256,0.8395323655710799>,0.05
    ,<-0.12417751781874967,-0.0006196970448052593,0.8786369095869186>,0.05
    ,<-0.13964327635180726,-0.0006730252273839026,0.9261730355845139>,0.05
    ,<-0.13465134550003074,-0.0006749698991329312,0.9759030785966725>,0.05
    ,<-0.11095329786158045,-0.0006276180888407798,1.0199033733185976>,0.05
    ,<-0.07534756316182889,-0.0005551440591313473,1.0549750219348453>,0.05
    ,<-0.03466157895774082,-0.0004801695291061785,1.084009228839247>,0.05
    ,<0.006442647343496824,-0.0004091294203078548,1.11245633893959>,0.05
    ,<0.0448189925603292,-0.00033815243342058017,1.1444926859510791>,0.05
    ,<0.07878475385065509,-0.0002620541429685085,1.1811751210813735>,0.05
    ,<0.10920142071283008,-0.00018071088224826844,1.2208533173886933>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
