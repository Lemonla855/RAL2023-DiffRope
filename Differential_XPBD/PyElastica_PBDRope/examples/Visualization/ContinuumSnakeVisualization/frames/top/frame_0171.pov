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
    ,<-0.1683876045301365,-0.0012857850796406767,4.216892069525379>,0.05
    ,<-0.12686905647715552,-0.001224250215315596,4.244752542734439>,0.05
    ,<-0.08742656760985759,-0.0011459613467832439,4.275485283657623>,0.05
    ,<-0.05323933388610538,-0.0010342363134950808,4.311974267373246>,0.05
    ,<-0.028261664663986982,-0.0008847900635819689,4.355287110762162>,0.05
    ,<-0.016308774924429753,-0.0007155765449341447,4.403830461192207>,0.05
    ,<-0.019750888067687506,-0.0005571417774222393,4.453701542853054>,0.05
    ,<-0.03860074595419829,-0.0004359502724065434,4.500000920478176>,0.05
    ,<-0.07041509821373275,-0.0003687998493886031,4.538564262926402>,0.05
    ,<-0.11091854960851145,-0.00035894975699511,4.56787853807401>,0.05
    ,<-0.15555821626954594,-0.000396013229889154,4.590411428223597>,0.05
    ,<-0.20063622248249774,-0.0004609884010241823,4.612060488785162>,0.05
    ,<-0.24220105351812846,-0.0005325963643857162,4.6398614792776485>,0.05
    ,<-0.27409386862504714,-0.0005846296994339811,4.6783710971245736>,0.05
    ,<-0.2891196964848684,-0.0005929193786717723,4.726055104371414>,0.05
    ,<-0.28410824665900813,-0.0005527256478158898,4.7757931748563225>,0.05
    ,<-0.26250722550031175,-0.00048091192073240204,4.820872045813989>,0.05
    ,<-0.23102734754926835,-0.00040446873298216087,4.8597018060443515>,0.05
    ,<-0.19555661505565908,-0.0003406198324775683,4.894928395204198>,0.05
    ,<-0.1598405184536673,-0.0002903592172745293,4.929911843506488>,0.05
    ,<-0.1253367222869021,-0.0002476289518152339,4.966096331088762>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
