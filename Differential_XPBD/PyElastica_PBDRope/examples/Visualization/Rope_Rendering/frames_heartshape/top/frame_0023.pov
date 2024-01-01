#include "../default.inc"

camera{
    location <0,200,3>
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
    b_spline 30
    ,<0.0,0.0,-7.5>,0.20090889930725098
    ,<-0.002933305921033025,0.2539643943309784,-7.291528224945068>,0.19874358177185059
    ,<-0.006099296268075705,0.5069192051887512,-7.083456039428711>,0.19885824620723724
    ,<-0.009746687486767769,0.7608848214149475,-6.873767852783203>,0.19886042177677155
    ,<-0.013961379416286945,1.011681079864502,-6.6658501625061035>,0.1988789141178131
    ,<-0.018846577033400536,1.2655528783798218,-6.455105781555176>,0.19890417158603668
    ,<-0.023956071585416794,1.514814853668213,-6.249677658081055>,0.19893786311149597
    ,<-0.028787333518266678,1.7718814611434937,-6.043248176574707>,0.19899767637252808
    ,<-0.0319799967110157,2.0268607139587402,-5.8498148918151855>,0.19904370605945587
    ,<-0.032354019582271576,2.297156572341919,-5.664677619934082>,0.1991240531206131
    ,<-0.02940094843506813,2.5731892585754395,-5.501893043518066>,0.1991153508424759
    ,<-0.024621738120913506,2.8629207611083984,-5.355982303619385>,0.19912050664424896
    ,<-0.023865515366196632,3.1590704917907715,-5.2090888023376465>,0.19894097745418549
    ,<-0.028096448630094528,3.427504539489746,-5.029808044433594>,0.1985374391078949
    ,<-0.03998899832367897,3.6498191356658936,-4.733569622039795>,0.19674761593341827
    ,<-0.030559919774532318,3.7477896213531494,-4.378549575805664>,0.19326910376548767
    ,<-0.06390201300382614,3.7412424087524414,-3.8550894260406494>,0.18703000247478485
    ,<-0.027639402076601982,3.625725269317627,-3.363171100616455>,0.18046967685222626
    ,<-0.04792983457446098,3.424382448196411,-2.635545492172241>,0.17532919347286224
    ,<-0.042847853153944016,3.231201410293579,-2.038327932357788>,0.1713876724243164
    ,<-0.037195682525634766,2.938307046890259,-1.1687006950378418>,0.16802214086055756
    ,<-0.034205302596092224,2.689290761947632,-0.47296860814094543>,0.1649564653635025
    ,<-0.0292670801281929,2.3360841274261475,0.5024141669273376>,0.16318506002426147
    ,<-0.025607267394661903,2.0503456592559814,1.2863044738769531>,0.16144119203090668
    ,<-0.021085992455482483,1.678582787513733,2.3076276779174805>,0.16091378033161163
    ,<-0.017426766455173492,1.3707655668258667,3.1575169563293457>,0.1600690484046936
    ,<-0.013076777569949627,1.0049962997436523,4.174021244049072>,0.1602019965648651
    ,<-0.009217883460223675,0.6837541460990906,5.073633193969727>,0.16002075374126434
    ,<-0.004958529490977526,0.33423304557800293,6.05753755569458>,0.15842655301094055
    ,<0.0,0.0,7.0>,0.17448189854621887
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
