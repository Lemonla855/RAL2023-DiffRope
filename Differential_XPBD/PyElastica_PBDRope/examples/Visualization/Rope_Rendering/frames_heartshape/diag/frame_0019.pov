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
    ,<0.0,0.0,-7.5>,0.2010631412267685
    ,<-0.02105732262134552,0.09911169111728668,-7.199581623077393>,0.199236661195755
    ,<-0.042162973433732986,0.1985839158296585,-6.898133754730225>,0.1993207484483719
    ,<-0.06304733455181122,0.2972427308559418,-6.599214553833008>,0.19932839274406433
    ,<-0.08408283442258835,0.39688199758529663,-6.297272205352783>,0.19935041666030884
    ,<-0.1046484187245369,0.4944997727870941,-6.001138687133789>,0.19939476251602173
    ,<-0.12547527253627777,0.5934085249900818,-5.700351715087891>,0.19948440790176392
    ,<-0.1455509066581726,0.6885635256767273,-5.409918785095215>,0.1996181309223175
    ,<-0.1660895049571991,0.785387396812439,-5.113466262817383>,0.1998317986726761
    ,<-0.18573974072933197,0.8771588802337646,-4.833380699157715>,0.20008398592472076
    ,<-0.2069658488035202,0.9750006198883057,-4.5432891845703125>,0.20036600530147552
    ,<-0.2283194214105606,1.0721943378448486,-4.27903413772583>,0.20056898891925812
    ,<-0.25749537348747253,1.2008575201034546,-3.993013381958008>,0.20052257180213928
    ,<-0.29167434573173523,1.3374202251434326,-3.752270221710205>,0.20008720457553864
    ,<-0.3462340831756592,1.5143249034881592,-3.438237428665161>,0.19864265620708466
    ,<-0.3694143295288086,1.5971641540527344,-3.1460394859313965>,0.19581842422485352
    ,<-0.3833664655685425,1.6142936944961548,-2.6398260593414307>,0.1908511221408844
    ,<-0.3946717381477356,1.6121746301651,-2.248844861984253>,0.18529736995697021
    ,<-0.3705357015132904,1.504577875137329,-1.5408787727355957>,0.17968207597732544
    ,<-0.3450983166694641,1.4168578386306763,-1.048351764678955>,0.17548124492168427
    ,<-0.30005592107772827,1.2592278718948364,-0.18492408096790314>,0.17243145406246185
    ,<-0.269266813993454,1.1495211124420166,0.4065248966217041>,0.1696641594171524
    ,<-0.2222772091627121,0.975338339805603,1.3529624938964844>,0.1687639206647873
    ,<-0.19104444980621338,0.8535389304161072,2.0245134830474854>,0.16751357913017273
    ,<-0.1497064232826233,0.6847917437553406,2.9704859256744385>,0.16798563301563263
    ,<-0.12028519809246063,0.5578567385673523,3.697056293487549>,0.16764481365680695
    ,<-0.08579748123884201,0.40338385105133057,4.594919681549072>,0.16855266690254211
    ,<-0.06275971233844757,0.2713641822338104,5.358803749084473>,0.16858932375907898
    ,<-0.028234299272298813,0.1341107189655304,6.201425552368164>,0.1680452823638916
    ,<0.0,0.0,7.0>,0.18011048436164856
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
