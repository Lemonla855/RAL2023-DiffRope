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
    ,<-0.08360206221735411,-0.0016507244304449463,2.8638272400772538>,0.05
    ,<-0.12258318348689927,-0.0014513924498494857,2.8951265714442735>,0.05
    ,<-0.15852860495076668,-0.0012378320483362956,2.9298717658614852>,0.05
    ,<-0.1875804754731384,-0.001001380149114489,2.970555553587238>,0.05
    ,<-0.20567019617854448,-0.0007500178012869469,3.0171564940350524>,0.05
    ,<-0.20971820986238696,-0.0005143760123571331,3.06697745855764>,0.05
    ,<-0.1987534814319529,-0.0003324761508214612,3.1157451858318113>,0.05
    ,<-0.17421493879817607,-0.0002292879349073415,3.1592970907089395>,0.05
    ,<-0.13940510790545912,-0.00021163015591692106,3.195184008876997>,0.05
    ,<-0.09853870499805982,-0.0002680090701725803,3.2239984005439077>,0.05
    ,<-0.055680505871901216,-0.00037317017463335225,3.2497653450454203>,0.05
    ,<-0.014855619518107103,-0.0004955437051707216,3.278642876969732>,0.05
    ,<0.018343761508056382,-0.0006053816277849771,3.3160265677648413>,0.05
    ,<0.0365256685867536,-0.0006719632675261815,3.362590771482584>,0.05
    ,<0.03438809311275998,-0.000671922318232145,3.4125252906375416>,0.05
    ,<0.013230018544584096,-0.0006069655884551675,3.4578032060671133>,0.05
    ,<-0.020431448091995395,-0.0005002653992064093,3.4947476628597514>,0.05
    ,<-0.05989892294382829,-0.00038046317562015714,3.525420288529868>,0.05
    ,<-0.10068640460505493,-0.0002645401991853109,3.55432425275639>,0.05
    ,<-0.14026686248254022,-0.0001553910170010346,3.5848673292461806>,0.05
    ,<-0.1780379136167668,-5.0440929731646534e-05,3.617626859759496>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
