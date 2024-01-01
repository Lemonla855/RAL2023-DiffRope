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
    ,<-0.020098848670972697,-0.0001326370930692585,0.00514459767950937>,0.05
    ,<-0.01943530727773262,-9.347819751613428e-05,0.055140007377983594>,0.05
    ,<-0.017168160723126717,-5.4424590098549024e-05,0.10509135322530105>,0.05
    ,<-0.012682971763562081,-1.787677621963785e-05,0.15489541124288234>,0.05
    ,<-0.006040213207049365,1.1403299456781021e-05,0.20446054366697308>,0.05
    ,<0.0022914176169960307,2.9394896183389668e-05,0.2537727188238699>,0.05
    ,<0.011589308690698769,3.547619134737295e-05,0.3029150658277391>,0.05
    ,<0.020943822085549617,3.237996643433003e-05,0.3520497941021773>,0.05
    ,<0.02917756573946004,2.544094886252873e-05,0.401387329395143>,0.05
    ,<0.03485414101325786,2.0927713522985608e-05,0.45108590519653385>,0.05
    ,<0.03660962670330172,2.330981220026099e-05,0.5010772784248586>,0.05
    ,<0.033630424556214907,3.231365950556468e-05,0.5510083556227032>,0.05
    ,<0.02604628505224855,4.1625202998030456e-05,0.6004445603703108>,0.05
    ,<0.01510086179922341,4.392087012870974e-05,0.6492448698321084>,0.05
    ,<0.0029308401325876176,3.6406775443391326e-05,0.6977568412971386>,0.05
    ,<-0.008022154075657931,2.0990640980678007e-05,0.7465605630477924>,0.05
    ,<-0.01567581650551426,2.392933492604376e-06,0.7959886124700511>,0.05
    ,<-0.018857701260966468,-1.4831133761008413e-05,0.845900673873174>,0.05
    ,<-0.017592453864503123,-2.8729558505364954e-05,0.8958938832818608>,0.05
    ,<-0.01296206907402357,-4.0533125446858074e-05,0.9456837545346106>,0.05
    ,<-0.006654135963800137,-5.192054261400036e-05,0.9952849565583177>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
