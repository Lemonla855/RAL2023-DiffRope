#include "../default.inc"

camera{
    location <40.0,100.5,-40.0>
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
    b_spline 30
    ,<0.0,0.0,-7.5>,0.1923915445804596
    ,<-0.060048289597034454,0.6305018067359924,-7.516176700592041>,0.1796739250421524
    ,<-0.1205841675400734,1.2607107162475586,-7.532844543457031>,0.18101656436920166
    ,<-0.1821134090423584,1.890421986579895,-7.550477981567383>,0.18090909719467163
    ,<-0.24519316852092743,2.5196993350982666,-7.569558620452881>,0.18094728887081146
    ,<-0.31049904227256775,3.1492481231689453,-7.590703964233398>,0.1809162199497223
    ,<-0.37894150614738464,3.7810428142547607,-7.615030288696289>,0.1807405799627304
    ,<-0.4518184959888458,4.419109344482422,-7.644841194152832>,0.1802576780319214
    ,<-0.5309181809425354,5.070026397705078,-7.684528350830078>,0.1792537122964859
    ,<-0.6183926463127136,5.742318153381348,-7.741027355194092>,0.1775515377521515
    ,<-0.7162020802497864,6.443802356719971,-7.822621822357178>,0.17520026862621307
    ,<-0.825080931186676,7.176537036895752,-7.935065746307373>,0.17269769310951233
    ,<-0.9432948231697083,7.930420398712158,-8.075190544128418>,0.171067014336586
    ,<-1.0657870769500732,8.678178787231445,-8.223708152770996>,0.17168331146240234
    ,<-1.1844323873519897,9.375353813171387,-8.339567184448242>,0.1759975105524063
    ,<-1.2896995544433594,9.966767311096191,-8.354366302490234>,0.1844022572040558
    ,<-1.3709008693695068,10.382221221923828,-8.148930549621582>,0.19000250101089478
    ,<-1.4017853736877441,10.47413158416748,-7.613005638122559>,0.17627650499343872
    ,<-1.3469880819320679,10.049798965454102,-6.7817912101745605>,0.1561451107263565
    ,<-1.2525426149368286,9.37757682800293,-5.755479335784912>,0.14824485778808594
    ,<-1.1299370527267456,8.523473739624023,-4.571901798248291>,0.14374172687530518
    ,<-0.991201639175415,7.55796480178833,-3.2851808071136475>,0.1419546753168106
    ,<-0.8472999930381775,6.545451641082764,-1.9493789672851562>,0.1416851133108139
    ,<-0.7062939405441284,5.5333147048950195,-0.6067453622817993>,0.14229492843151093
    ,<-0.5725667476654053,4.547611713409424,0.7173769474029541>,0.1432594209909439
    ,<-0.44723594188690186,3.595862627029419,2.013428211212158>,0.14422118663787842
    ,<-0.3292812705039978,2.673866033554077,3.282763957977295>,0.14485974609851837
    ,<-0.21674844622612,1.7727491855621338,4.532015800476074>,0.14586247503757477
    ,<-0.10758699476718903,0.8837355375289917,5.768857002258301>,0.1425495147705078
    ,<0.0,0.0,7.0>,0.16556265950202942
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
