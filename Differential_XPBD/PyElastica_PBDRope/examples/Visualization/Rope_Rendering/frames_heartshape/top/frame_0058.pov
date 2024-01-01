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
    linear_spline 30
    ,<0.0,0.0,-30.0>,0.5567131042480469
    ,<-0.28902772068977356,1.4223859310150146,-28.622875213623047>,0.5422320365905762
    ,<-0.758285403251648,2.6287002563476562,-27.097145080566406>,0.5484432578086853
    ,<-1.1166396141052246,4.195852756500244,-25.905725479125977>,0.5457224249839783
    ,<-1.614733338356018,4.1178412437438965,-23.969440460205078>,0.5468239188194275
    ,<-2.2255101203918457,5.682536602020264,-22.878400802612305>,0.5462617874145508
    ,<-1.8977766036987305,7.34859037399292,-21.817354202270508>,0.5465637445449829
    ,<-3.434670925140381,6.971249103546143,-20.592374801635742>,0.5465907454490662
    ,<-4.8765435218811035,7.276190757751465,-19.24246597290039>,0.5467454195022583
    ,<-5.28558874130249,8.241385459899902,-17.541669845581055>,0.5467561483383179
    ,<-6.645345687866211,6.8426737785339355,-17.96881866455078>,0.5467838644981384
    ,<-8.168112754821777,5.584580898284912,-18.265819549560547>,0.5467183589935303
    ,<-9.860679626464844,4.525300979614258,-18.17804527282715>,0.5466510653495789
    ,<-11.552804946899414,3.4615132808685303,-18.091472625732422>,0.5464045405387878
    ,<-12.780558586120605,2.017930746078491,-18.741971969604492>,0.5468491315841675
    ,<-14.761985778808594,1.7939549684524536,-18.554101943969727>,0.5448209643363953
    ,<-15.483428001403809,3.0839874744415283,-19.845983505249023>,0.551114559173584
    ,<-13.61657428741455,3.6367733478546143,-18.710359573364258>,0.49940019845962524
    ,<-13.18232250213623,3.311493158340454,-15.418875694274902>,0.4568042457103729
    ,<-12.426911354064941,2.8840222358703613,-11.439270973205566>,0.46137571334838867
    ,<-11.330452919006348,2.5068185329437256,-7.274204730987549>,0.4593539237976074
    ,<-10.059619903564453,2.2018556594848633,-3.196045398712158>,0.464015930891037
    ,<-8.760703086853027,1.9351999759674072,0.7719140648841858>,0.46494704484939575
    ,<-7.483360290527344,1.672467827796936,4.684343338012695>,0.46600085496902466
    ,<-6.303758144378662,1.5025509595870972,8.596222877502441>,0.46622586250305176
    ,<-4.9744415283203125,1.1391727924346924,12.440986633300781>,0.4658358097076416
    ,<-3.7267441749572754,0.856830894947052,16.33062171936035>,0.4660264551639557
    ,<-2.4836251735687256,0.5724384188652039,20.221094131469727>,0.4652039408683777
    ,<-1.2423889636993408,0.28765174746513367,24.111221313476562>,0.46843650937080383
    ,<0.0,0.0,28.0>,0.456595778465271
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }