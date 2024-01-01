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
    ,<0.0,0.0,-7.5>,0.19928471744060516
    ,<0.005532151088118553,0.37552112340927124,-7.362489223480225>,0.19484150409698486
    ,<0.011111237108707428,0.7474417686462402,-7.226160049438477>,0.19504518806934357
    ,<0.01706365868449211,1.1309177875518799,-7.085690975189209>,0.1948474496603012
    ,<0.02305203303694725,1.5021973848342896,-6.950649738311768>,0.1946462094783783
    ,<0.02960539422929287,1.9003266096115112,-6.809021949768066>,0.1943429410457611
    ,<0.03558725863695145,2.2741925716400146,-6.681509017944336>,0.19402503967285156
    ,<0.04134465381503105,2.693530797958374,-6.548829078674316>,0.19370199739933014
    ,<0.044908031821250916,3.072909355163574,-6.440478801727295>,0.1933884173631668
    ,<0.04575912654399872,3.5111887454986572,-6.330821990966797>,0.19330614805221558
    ,<0.042730432003736496,3.8947741985321045,-6.246335029602051>,0.19324713945388794
    ,<0.03415429964661598,4.330150127410889,-6.154355049133301>,0.1937488168478012
    ,<0.02187824621796608,4.708310127258301,-6.060290813446045>,0.19415268301963806
    ,<0.005835163872689009,5.092550754547119,-5.914104461669922>,0.19496920704841614
    ,<-0.008016008883714676,5.412072658538818,-5.665580749511719>,0.19429592788219452
    ,<-0.0103067047894001,5.614933490753174,-5.31527853012085>,0.19147583842277527
    ,<-0.025029191747307777,5.668130397796631,-4.754903793334961>,0.18459290266036987
    ,<0.011032942682504654,5.592970848083496,-4.221426963806152>,0.1773964762687683
    ,<-0.031987328082323074,5.333855628967285,-3.417349100112915>,0.17132236063480377
    ,<-0.01662290468811989,5.042816162109375,-2.812870502471924>,0.16684560477733612
    ,<-0.010674591176211834,4.578502178192139,-1.849646806716919>,0.1629350483417511
    ,<-0.0056941453367471695,4.2126264572143555,-1.129690408706665>,0.1596662551164627
    ,<0.0020896978676319122,3.6450812816619873,-0.04069651663303375>,0.15778380632400513
    ,<0.007015830371528864,3.21353816986084,0.7768782377243042>,0.15596026182174683
    ,<0.011782152578234673,2.612523078918457,1.9162498712539673>,0.1554514616727829
    ,<0.013219952583312988,2.1428515911102295,2.8128106594085693>,0.15458761155605316
    ,<0.012563151307404041,1.5581982135772705,3.9394283294677734>,0.15478770434856415
    ,<0.0096283545717597,1.0654088258743286,4.9006242752075195>,0.15466591715812683
    ,<0.004786748439073563,0.5165855884552002,5.979578495025635>,0.15266135334968567
    ,<0.0,0.0,7.0>,0.17105929553508759
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
