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
    ,<0.07342600043401205,-0.0014749504696189109,3.318744722651463>,0.05
    ,<0.03762538309608598,-0.0012840354823113913,3.3536408014194885>,0.05
    ,<-0.0006099807041709573,-0.0011077397534289422,3.3858580863819503>,0.05
    ,<-0.04140777213376518,-0.0009578038138566656,3.414770582668006>,0.05
    ,<-0.08368561281604668,-0.0008385086472725799,3.441482023274792>,0.05
    ,<-0.12595597326569416,-0.0007419020831448113,3.4682124229484153>,0.05
    ,<-0.16644590024679104,-0.0006509977385306721,3.4975724084837787>,0.05
    ,<-0.20267467682550744,-0.000545548774734284,3.5320503520423188>,0.05
    ,<-0.2307954993784657,-0.0004102623221225793,3.573401231353701>,0.05
    ,<-0.24554311942779364,-0.00024619644999709705,3.6211739866943518>,0.05
    ,<-0.24228237595824978,-8.195311093701174e-05,3.671053228914848>,0.05
    ,<-0.22053577753988776,4.075667512385903e-05,3.716054282607233>,0.05
    ,<-0.18510036833310758,9.406771197979347e-05,3.751303378385963>,0.05
    ,<-0.14273510730054217,7.28885281448984e-05,3.7778348389981287>,0.05
    ,<-0.09878164379573237,-8.789627611758662e-06,3.8016524769261837>,0.05
    ,<-0.057489696804671406,-0.0001280176753462331,3.8298340367646735>,0.05
    ,<-0.02363459598572683,-0.0002622247313745102,3.866614674666821>,0.05
    ,<-0.0014874074135566976,-0.0003960679886987237,3.9114279398389513>,0.05
    ,<0.00804714289616145,-0.0005246896611332156,3.9604967974788843>,0.05
    ,<0.008113934725885032,-0.0006515129295561756,4.010484667853456>,0.05
    ,<0.0037890401438896376,-0.0007789194957748717,4.060288541854586>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }