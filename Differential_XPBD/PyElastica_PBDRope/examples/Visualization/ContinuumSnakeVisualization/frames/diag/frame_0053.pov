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
    ,<0.039435709974356255,-0.0020141313554171134,0.9710276112985593>,0.05
    ,<0.06862645948847199,-0.0016789525970349814,1.0116045855826157>,0.05
    ,<0.09338403273935991,-0.0013316203443718463,1.0550306937034435>,0.05
    ,<0.10922551415993313,-0.0009697620986797721,1.1024420734886093>,0.05
    ,<0.11262624494568585,-0.0006100056102269464,1.1523139343259787>,0.05
    ,<0.10207578742961161,-0.0002902232546397145,1.2011753596172094>,0.05
    ,<0.0784541463636504,-5.120063450380572e-05,1.245233322421051>,0.05
    ,<0.044501197019567444,8.395693151413462e-05,1.2819334527688893>,0.05
    ,<0.0038214668512446615,0.0001145118354497914,1.3110116948295525>,0.05
    ,<-0.03999339376641752,5.840554226831959e-05,1.3351179934531108>,0.05
    ,<-0.08365815353494449,-5.3918354586307386e-05,1.3594974067558436>,0.05
    ,<-0.12309158254088183,-0.00018601374989836642,1.3902456922702386>,0.05
    ,<-0.15192813923378629,-0.00030534983400483086,1.4310856845021964>,0.05
    ,<-0.16297434958913595,-0.00038150716459206194,1.4798355477869731>,0.05
    ,<-0.15324645581717922,-0.000395940479757443,1.5288594606049974>,0.05
    ,<-0.1266846750477091,-0.0003571024655105954,1.5711957860088446>,0.05
    ,<-0.09053507073045823,-0.00029061882955854493,1.6057125363418037>,0.05
    ,<-0.05086941122281713,-0.0002171726698530197,1.6361320372375936>,0.05
    ,<-0.01161429637448991,-0.00014289973693802256,1.667086362454947>,0.05
    ,<0.025024201875175615,-6.710082009308612e-05,1.7011021228378438>,0.05
    ,<0.0590192287050445,1.0649166335153407e-05,1.737763864126755>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
