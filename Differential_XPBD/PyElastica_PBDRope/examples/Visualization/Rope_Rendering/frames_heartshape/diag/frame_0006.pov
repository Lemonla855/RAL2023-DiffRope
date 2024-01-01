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
    ,<0.0,0.0,-7.5>,0.2012176662683487
    ,<-0.016439899802207947,0.13030610978603363,-7.218109130859375>,0.1995704472064972
    ,<-0.03289911523461342,0.2606427073478699,-6.936153411865234>,0.19963903725147247
    ,<-0.04939674213528633,0.39102038741111755,-6.654080390930176>,0.19962741434574127
    ,<-0.06595098227262497,0.5214053392410278,-6.371870517730713>,0.1996191143989563
    ,<-0.08257703483104706,0.6516835689544678,-6.089589595794678>,0.19961556792259216
    ,<-0.0992826297879219,0.7816176414489746,-5.807492733001709>,0.1996278464794159
    ,<-0.11606085300445557,0.9108294248580933,-5.526211738586426>,0.19967661798000336
    ,<-0.13288339972496033,1.0388882160186768,-5.247008800506592>,0.19979454576969147
    ,<-0.14970333874225616,1.1656482219696045,-4.9719953536987305>,0.20001976191997528
    ,<-0.1664796620607376,1.2919989824295044,-4.704098701477051>,0.20037205517292023
    ,<-0.1832180619239807,1.421047329902649,-4.446487903594971>,0.20081022381782532
    ,<-0.19995686411857605,1.559030532836914,-4.201106548309326>,0.20118539035320282
    ,<-0.21655310690402985,1.7135404348373413,-3.9654343128204346>,0.2012150138616562
    ,<-0.23246264457702637,1.8880935907363892,-3.727360486984253>,0.20054522156715393
    ,<-0.24739256501197815,2.0629324913024902,-3.453468084335327>,0.19854645431041718
    ,<-0.25748226046562195,2.185429096221924,-3.0887246131896973>,0.19354350864887238
    ,<-0.2537858784198761,2.2090070247650146,-2.5972392559051514>,0.18480360507965088
    ,<-0.24268995225429535,2.072964668273926,-1.9902987480163574>,0.17621344327926636
    ,<-0.2286546528339386,1.899677038192749,-1.2852494716644287>,0.1705751121044159
    ,<-0.21192894876003265,1.7040432691574097,-0.5029058456420898>,0.16707834601402283
    ,<-0.19277521967887878,1.4989584684371948,0.3294060230255127>,0.16549460589885712
    ,<-0.17157351970672607,1.2939770221710205,1.1852140426635742>,0.1652512401342392
    ,<-0.14879700541496277,1.0945415496826172,2.0443222522735596>,0.16578145325183868
    ,<-0.12493102997541428,0.9023266434669495,2.894976854324341>,0.16659361124038696
    ,<-0.10039188712835312,0.7164527177810669,3.733020782470703>,0.16735349595546722
    ,<-0.07548157125711441,0.5349369645118713,4.559340476989746>,0.1678895652294159
    ,<-0.05038703233003616,0.35582372546195984,5.377087593078613>,0.16834644973278046
    ,<-0.025209274142980576,0.1777496635913849,6.189693927764893>,0.16723036766052246
    ,<0.0,0.0,7.0>,0.17979663610458374
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
