// 获取地址栏参数
function getParam(key){
    var reg = new RegExp("(^|&)"+key+"=([^&]*)(&|$)");
    var result = window.location.search.substr(1).match(reg);
    return result?decodeURIComponent(result[2]):null;
}
//判断是否支持本地缓存
function isLocalStorage() {
    var testKey = 'test',
        storage = window.localStorage;
    try {
        storage.setItem(testKey, 'testValue');
        storage.removeItem(testKey);
        return true;
    } catch (error) {
        return false;
    }
}
// 时间戳转换为日期
function formatDate(time) {
    var d=new Date(time*1000);
    var year=d.getFullYear();
    var month=d.getMonth()+1;
    month = month<10?"0"+month:month;
    var date=d.getDate();
    date = date<10?"0"+date:date;
    return year+"/"+month+"/"+date;
}
// 设置本地存储
function setLocalStorage(lname,lvalue){
	localStorage.setItem(lname, JSON.stringify(lvalue));
}
// 获取本地存储
function getLocalStorage(lname){
	return JSON.parse(localStorage.getItem(lname));
}
//设置cookie
function setCookie(cname,cvalue,exdays){
    var d = new Date();
    d.setTime(d.getTime() + (exdays*12*60*60*1000));
    var expires = "expires="+d.toUTCString();
    document.cookie = cname + "=" + cvalue + "; " + expires+";path=/"; 
}
//获取cookie
function getCookie(cname){
    var name = cname + "=";
    var ca = document.cookie.split(';');
    for(var i=0; i<ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0)==' ') c = c.substring(1);
        if (c.indexOf(name) != -1) return c.substring(name.length, c.length);
    }
    return "";
}
new FastClick(document.body);
//点击关闭按钮关闭下载banner
$(".icon_close").click(function(){
	$(".download").hide();
	setCookie("download","1",2);
})
var downloadStatus = getCookie("download");
if (downloadStatus =="1") {
	$(".download").hide();
}else{
	$(".download").show();
};

var status = getCookie("apicustomer_id") || '';//判断用户是否登录
var user_mobile = getCookie("user_mobile");//用户手机号
var room_id = getParam("room_id");//房间id
var city_code = getParam("city_code");//城市编号
var city_name = getParam("city_name");//城市名称
var appoint = getParam("appoint") || '';
var callpage = getParam("callpage") || '';
var come = getParam("come") || '';//判断页面来源以定位返回页面
var app = getParam("infofrom") || '';//判断从app来
var room_longitude = "";
var room_latitude = "";
var room_no = "";//房源编号
var ld_number = "";//房东手机号
var owner_id = "";//房东ID
var owner_name = "";//房东名字
var has_appointed = "";//租客是否预约过
var alert_text_cannot = "";//预约返回字段
var userNumber = "";//是否优先使用虚拟号拨打
var number_id = "";//房源虚拟号码id
var virtual_flag = '';//电话统计
var is_owner_type='';//房东类型
var is_regroup_bool='';//聚合


var url = window.location.href;
var u = navigator.userAgent;
var isiOS = !!u.match(/\(i[^;]+;( U;)? CPU.+Mac OS X/); //ios终端

var host = window.location.host;
var http ="http://"+host+"/Home/House/";
// var http = "http://127.0.0.1/gaoduwap/Home/House/";
//点击头部返回 定位返回页面
$("#backPage").click(function(){
	if (appoint != '1' && callpage != '1' && app != 'storeapp') {
		history.go(-1);
	}else{
		window.location.href = 'http://'+window.location.host+'/'+city_name+'/roomlist.html'+window.location.hash;
	};
})
// 得到配置文件
function getConfig(){
	var configData;
	if (isLocalStorage()) {
		var configCache = getLocalStorage("configData"+city_code);
		if (!configCache) {
			$.ajax({
				type:"GET",
				async:false,
				dataType:'json',
				url:http+"config.html",
				data:{city_code:city_code},
				success:function(data){
					setLocalStorage("configData"+city_code,data);
					configData = data;
				}
			})
			return configData;
		}else{
			return configCache;
		}
	}else{
		$.ajax({
			type:"GET",
			async:false,
			dataType:'json',
			url:http+"config.html",
			data:{city_code:city_code},
			success:function(data){
				configData = data;
			}
		})
		return configData;
	};
}
// 获取数据
function getDate(){
	$.ajax({
		type:"GET",
		async:false,
		dataType:'json',
		url:http+"housedetail.html",
		data:{room_id:room_id,city_code:city_code},
		success:function(res){
			if (res.status == 200) {
				var data = res.data;//数据内容
				is_owner_type=data.is_owner;
				is_regroup_bool=data.is_regroup;
				room_longitude = data.room_longitude;//经度
				room_latitude = data.room_latitude;//纬度
				room_no = data.room_no;//房源编号
				ld_number = data.owne_number_txt;//房东电话
				owner_id = data.owner_id;//房东id
				owner_name = data.owner_name;//房东名字
				has_appointed = data.has_appointed;//是否预约
				alert_text_cannot = data.alert_text_cannot;//预约返回字段
				userNumber = data.use_virtual_number;//是否优先使用虚拟号拨打电话
				number_id = data.virtual_number_id;//虚拟号id
				console.log(res)
				var configMess = getConfig();//得到配置文件
				console.log(configMess)
				var configData = configMess.data;//配置文件数据
                //详情页标题
				$("#title").html(data.estate_name);
				//循环配置文件数据，匹配数据
				var brand_type = "";
				for (var i = 0; i < configData.length; i++) {
					if(configData[i].type_no == data.room_direction){//房屋朝向
						var room_direction = configData[i].name;
						if (room_direction.length == 1) {
							room_direction = "朝"+room_direction;
						};
					};
					if (configData[i].type_no == data.brand_type) {//公寓类型
						brand_type = configData[i].name;
					};
					if(configData[i].type_no == data.pay_method){//付款方式
						var pay_method = configData[i].name;
					};
					if(configData[i].type_no == data.business_type){//小区住宅，品牌公寓，酒店
						var business_type = configData[i].name;
					};
					if(configData[i].type_no == data.room_type){//房屋类型
						var room_type = configData[i].name;
					};
					if(configData[i].type_no == data.decoration){//装修类型
						var decoration = configData[i].name;
					}
				};
				// 轮播图片
				var imgUrls = data.image_urls;//获取图片
				var videohtml = "";
				if (data.videos.length != 0) {
					videohtml += '<img class="play" src="../Public/images/play.png">';
					videohtml += '<div class="swiper-slide swiper-slide2"><img src='+data.videos[0].video_image_url+'><video controls="controls" preload="true"><source src='+data.videos[0].video_url+' type="video/mp4"></video></div></div>';
					$(".swiper-wrapper").append(videohtml);
				}
				if (imgUrls.length > 0) {
					var shtml = "";
					var bightml = "";
					for (var i = 0; i < imgUrls.length; i++) {
						// var point_index = imgUrls[i].lastIndexOf(".");
						// var corp_img_url=imgUrls[i].substring(0,point_index)+"_375_200"+imgUrls[i].substring(point_index);
						shtml += '<div class="swiper-slide"><img class="swiper-lazy" data-src='+imgUrls[i]+'><div class="swiper-lazy-preloader swiper-lazy-preloader-white"></div></div>';
						bightml += '<div class="swiper-slide"><div class="pinch-zoom"><img  class="swiper-lazy" data-src='+imgUrls[i]+'><div class="swiper-lazy-preloader swiper-lazy-preloader-white"></div></div></div>';
					};
					$("#swiper-wrapper1").append(shtml);
					$("#swiper-wrapper2").append(bightml);
				}else{
					$("#swiper-container1").empty().append("<img class='defaultImg' src='../Public/images/bg_default.png' >")
				};
				$(document).on("click","#swiper-wrapper1 .swiper-slide img",function(){
					$(".bigPic").css("visibility","visible");//显示大图
					$("body,html").css({"height":"100%","overflow":"hidden"});
				});
				$(document).on("click","#swiper-wrapper1 .play",function(){
					$("#swiper-container1 video").css("display","block");//显示大图
					$("#swiper-container1 video")[0].play();
					$(this).hide();
				});
				$(document).on("click",".bigPic .play",function(){
					$(".bigPic video").css("display","block");//显示大图
					$(".bigPic video")[0].play();
					$(this).hide();
				});
				// 房间详情
				var rent_type = data.rent_type;//房屋类型
				if (rent_type == 2) {
					rent_type = "整租";
				}else if(rent_type == 1){
					rent_type = "合租";
				};
				var estate_name = data.estate_name;//小区名称
				var room_num = data.room_num;//几居室
				var room_money = data.room_money;//价格
				var room_name = data.room_name;//主卧
				var room_area = data.room_area;//面积
				var publicMain = "";//公用设施
				var privateMain = "";//私用设施
				var pubdata = data.public_facility;
				var privadata = data.room_facility;
				//房源信息内容加载
				var mesDetailmain = "";
				mesDetailmain += '<p class="house_title cf"><span>'+rent_type+'&nbsp;·&nbsp;</span><span>'+estate_name+'&nbsp;·&nbsp;</span><span>'+room_num+'居室</span></p>';
				if (data.is_owner == '5' || data.is_regroup == '1') {
					mesDetailmain += '<p class="house_text"><span class="price"><em>'+room_money+'</em> 元/月起</span>';
				}else{
					mesDetailmain += '<p class="house_text"><span class="price"><em>'+room_money+'</em> 元/月</span>';
				};
				mesDetailmain += '<span>'+room_name+'</span><span>'+room_area+'㎡</span><span>'+room_direction+'</span></p>';
				mesDetailmain += '<p class="house_label">';
				if (data.new_online == 1) {
					html += '<span>新上</span>';
				};
				if (data.tags) {
					var tagsLen = data.tags.length;
					for(var j=0;j<tagsLen;j++){
						mesDetailmain += '<span>'+data.tags[j]+'</span>';
					};
				};
				if (brand_type) {
					mesDetailmain += '<span>'+brand_type+'</span>';
				};
				if (data.pay_method) {
					mesDetailmain += '<span>'+pay_method+'</span>';
				};
				if (data.business_type && business_type != "小区住宅") {
					mesDetailmain += '<span>'+business_type+'</span>';
				};
				if (room_type.indexOf("隔断")>0) {
					mesDetailmain += '<span>隔断房</span>';
				};
				if (privadata.length != 0) {
					for (var i = 0; i < privadata.length; i++) {
						if(privadata[i].name.indexOf("卫") > -1){
							mesDetailmain += '<span>独卫</span>';
						};
						if(privadata[i].name.indexOf("厨") > -1){
							mesDetailmain += '<span>独厨</span>';
						};
					};
				};
				if (data.only_girl=="1") {
					mesDetailmain += '<span>限女生</span>';
				};
				if (data.decoration) {
					mesDetailmain += '<span>'+decoration+'</span>';
				};
				mesDetailmain += '</p>';
				$(".mesDetail").append(mesDetailmain);//房间信息头部
				//地图
				var maphtml = "";
				maphtml += '<a href="http://'+window.location.host+'/Home/House/mappositioning.html?region_name='+data.region_name+'&estate_name='+data.estate_name+'&address='+data.address+'&scope_name='+data.scope_name+'&latitude='+room_latitude+'&longitude='+room_longitude+'"><img src="http://api.map.baidu.com/staticimage/v2?ak=Le7RjXqw3KAKI9KUZys1SA5s&mcode=com.gaodu.loulifang&center='+room_longitude+','+room_latitude+'&markers='+room_longitude+','+room_latitude+'&markerStyles=-1,http://img.loulifang.com.cn/fang/F3/FA/f3da68ff17b54f70ba566c8dea7b2efa.png,-1,23,25&zoom=18&copyright=1&width=750&height=320"></a>';
				$(".map").append(maphtml);//地图
				var traffichtml = "";
				traffichtml += '<li><span><img src="../Public/images/xiaoq.png"></span><p>'+estate_name+'</p></li>';
				traffichtml += '<li class="dizhi"><span><img src="../Public/images/traff.png"></span><p>'+data.region_name+'－'+data.scope_name+'－'+data.address+'</p></li>';
				if (data.subway_objs.length != 0) {
					traffichtml += '<li><span><img src="../Public/images/subway.png"></span><p>距'+data.subway_objs[0].subwayline_name+data.subway_objs[0].subway_name+'站'+data.subway_objs[0].subway_distance+'米</p></li>';
				};
				$("#traffic").append(traffichtml);
				// 房间信息
				var housemess = "";
				housemess +=  "<li><span>面积</span><p>"+data.room_area+"/"+data.area+"㎡</p></li>";
				housemess +=  "<li><span>户型</span><p>"+data.room_num+"室"+data.hall_num+"厅"+data.wei_num+"卫</p></li>";
				if (data.floor != "0" && data.floor_total != "0") {
					housemess +=  "<li><span>楼层</span><p>"+data.floor+"/"+data.floor_total+"层</p></li>";
				};
				housemess +=  "<li><span>装修</span><p>"+decoration+"</p></li>";
				housemess +=  "<li><span>编号</span><p>"+data.room_no+"</p></li>";
				housemess +=  "<li><span>更新时间</span><p>"+formatDate(data.update_time)+"</p></li>";
				$("#houseMess").append(housemess);//房间信息数据
				// 公用设施
				if (pubdata.length != 0) {
					for (var i = 0; i < pubdata.length; i++) {
						if (pubdata[i].name.indexOf("卫") > -1) {
							pubdata[i].name = "卫生间";
						};
						publicMain += "<li><img src="+pubdata[i].img_url_bright+"><span>"+pubdata[i].name+"</span></li>";
					};
					$("#public").append(publicMain);//公用设施
				}else{
					$(".public").hide();
				};
				//独用设施
				if (privadata.length != 0) {
					for (var i = 0; i < privadata.length; i++) {
						if (privadata[i].name.indexOf("卫") > -1) {
							privadata[i].name = "卫生间";
						};
						if (privadata[i].name.indexOf("厨") > -1) {
							privadata[i].name = "厨房";
						};
						if (privadata[i].name.indexOf("阳台") > -1) {
							privadata[i].name = "阳台";
						};
						privateMain += "<li><img src="+privadata[i].img_url_bright+"><span>"+privadata[i].name+"</span></li>";
					};
					$("#private").append(privateMain);//私用设施
				}else{
					$(".private").hide();
				};
				// 隔壁房间
				var rooms = data.roomies;
				var roomsMain = "";
				if (rooms.length != 0) {
					roomsMain += '<h3 class="common_title">';
					if (data.business_type && business_type != "小区住宅") {
						roomsMain += '其他房型';
					};
					if (business_type == "小区住宅") {
						roomsMain += '隔壁房间';
					};
					roomsMain += '</h3><div class="houseIntro">';
					for (var i = 0; i < rooms.length; i++) {
						for (var j = 0; j < configData.length; j++) {
							if(configData[j].type_no == rooms[i].room_direction){//房屋朝向
								var room_direction2 = configData[j].name;
								if (room_direction2.length == 1) {
									room_direction2 = "朝"+room_direction2;
								};
							};
						}
						roomsMain += '<a class="houselist cf" href="http://'+window.location.host+'/'+city_name+'/roomDetail.html?room_id='+rooms[i].room_id+'&city_code='+city_code+'&city_name='+city_name+'"><div class="houseIntro_left fl"><img src='+rooms[i].main_img_path+'></div>';
						roomsMain += '<div class="fl houseIntro_right"><p class="rent_title"><span>'+rooms[i].room_name+'&nbsp;·&nbsp;</span><span>'+rooms[i].room_area+'㎡&nbsp;·&nbsp;</span><span>'+room_direction2+"</span></p>";
						if (rooms[i].status == 2) {
							roomsMain += '<p class="rent_price1"><em>'+rooms[i].room_money+'</em>&nbsp;元/月</p>';
							roomsMain += '<span class="rent_status1">待出租</span>';
						}else{
							roomsMain += '<p class="rent_price2"><em>'+rooms[i].room_money+'</em>&nbsp;元/月</p>';
							roomsMain += '<span class="rent_status2">已出租</span>';
						};
						roomsMain += '<p class="rent_label">';
						if (rooms[i].rent_type == '1') {
							roomsMain += '<span>合租</span>';
						}else if(rooms[i].rent_type == '2'){
							roomsMain += '<span>整租</span>';
						};
						roomsMain += '<span>'+pay_method+'</span></p></div></a></div>';
					};
					$("#roommate").append(roomsMain);//隔壁房间
				};
				var is_owner = data.is_owner;//是否是中介
				// 房东信息
				if (is_owner != '5' && data.is_regroup != '1') {
					var renterhtml = "";
					renterhtml += '<a class="renterTop cf" href="http://'+window.location.host+'/'+city_name+'/ownerInfo/'+data.owner_id+'/'+data.room_id+'.html"><div class="renterTop_l fl">';
					if (data.owner_avatar) {
						renterhtml += '<img src='+data.owner_avatar+'>';
					};
					renterhtml += '</div><div class="renterTop_r fl"><p>'+data.owner_name+'</p><span>已有评价（'+data.owner_evaluate_cnt+'）</span></div></a>';
					//房屋描述
					if(data.room_description){
						if (data.room_description.length > 48) {
							var room_description = data.room_description.substring(0,45)+'...';
							renterhtml += '<div class="renterCenter"><p>'+room_description+'</p><img src="../Public/images/xl.png"></div>';
						}else{
							renterhtml += '<div class="renterCenter"><p>'+data.room_description+'</p></div>';
						};
					}
					// 点击图片展开内容
					$(document).on("click",".renterCenter img",function(){
						if($(this).attr("src") == '../Public/images/xl.png'){
							$(this).attr("src","../Public/images/sl.png");
							$(".renterCenter p").html(data.room_description);
						}else{
							$(this).attr("src","../Public/images/xl.png");
							$(".renterCenter p").html(room_description);
						}
					})
					// 我希望租客能
					if (data.owner_like.length != 0) {
						renterhtml += '<div class="renterBottom"><p class="renterBottom_title">我希望租客能</p><div class="renterBottom_main cf">';
						for (var i = 0; i < data.owner_like.length; i++) {
							for (var j = 0; j < configData.length; j++) {
								if(configData[j].type_no == data.owner_like[i]){//房屋朝向
									var ovnerData = configData[j].name;
								};
							}
							renterhtml += '<span>'+ovnerData+'</span>';
						};
						renterhtml += '</div></div>';
					};
					$("#renter").append(renterhtml);//房东信息结束
				}else{
					$("#renter").hide();
				};
				// 报价
				if (is_owner == '5' || data.is_regroup == '1') {
					var zjhtml = "";
					var agents = data.agents;//中介报价信息
						console.log(agents);
					for (var i = 0; i < agents.length; i++) {
						zjhtml += '<div class="offer_main"><div class="offer_list"><div class="main1"><img src='+agents[i].agent_img_url+'></div>';
						if(agents[i].is_owner==4){
							zjhtml += '<div class="main2"><p>房东直租</p><span>'+agents[i].agent_name+'</span></div>';
						}else{
							zjhtml += '<div class="main2"><p>'+agents[i].agent_organization+'</p><span>'+agents[i].agent_name+'</span></div>';
						}

						zjhtml += '<div class="main3"><p>房租</p><span><em>'+agents[i].room_money+'</em>元/月</span></div>';
						zjhtml += '<div class="main4"><p>中介费</p><span>'+agents[i].agent_money+'&nbsp;元</span></div>';
						zjhtml += '<a href="javascript:;" class="main5" index='+agents[i].use_virtual_number+' phone_number='+agents[i].phone_number+' phone_numbers='+agents[i].phone_numbers+' data_id='+agents[i].virtual_number_id+'><img src="../Public/images/tel.png"><span>咨询</span></a></div></div>';
					};
					if (data.total_agent > agents.length) {
						zjhtml += '<a class="seeMore" href="http://'+window.location.host+'/Home/House/offer.html?room_id='+data.room_id+'&city_code='+city_code+'&city_name='+city_name+'&room_no='+room_no+'&ld_number='+ld_number+'&owner_id='+owner_id+'&owner_name='+owner_name+window.location.hash+'">查看全部'+data.total_agent+'个报价</a>';
					};
					$(".offer").append(zjhtml);
				}else{
					$(".offer").hide();//隐藏报价块
				};
				// 点击咨询
				$(document).on('click','.main5',function(){
					var use_number = $(this).attr("index");
					var phone_number = $(this).attr("phone_number");
					var virtual_number_id = $(this).attr("data_id");
					var phone_numbers = $(this).attr("phone_numbers");
					console.log(status);
					console.log(use_number);
					var status = getCookie("apicustomer_id") || '';//判断用户是否登录
					if (status) {
						//判断是否优先使用虚拟号拨打电话
						if (use_number == "1") {
							$.ajax({
								type:"GET",
								async:false,
								dataType:'json',
								url:http+"virtualcallno.html",
								data:{phone_number:user_mobile,virtual_number_id:virtual_number_id,room_id:room_id,room_no:room_no,city_code:city_code},
								success:function(data){
									console.log(data)
									var phone = data.data.virtual_numbers;//得到虚拟号
									virtual_flag = data.data.virtual_flag;//虚拟号码标记
									ld_number = phone;
									if(isiOS){
										window.location.href = 'tel:'+phone;
										call();
									}else{
										if (phone.substr(0,3) == '400') {
											var phone_qian = phone.substr(0,10);
											var phone_hou = phone.substr(11);
											$(".call").show();
											$("#tel").html("请在拨通"+phone_qian+"后<br/>再输入"+phone_hou);
											$("#call").click(function(){
												
												window.location.href = 'tel:'+phone;
												call();
											})
										}else{
											window.location.href = 'tel:'+phone;
												call();
										};
									}
									// $(".call").show();
									// $("#tel").html(phone);
									// $("#call").attr("href","tel:"+phone);
								}
							})
						}else{
							virtual_flag = -1;//非虚拟号码
							var qian = phone_numbers.substr(0,10);
							var hou = phone_numbers.substr(11);
							if (isiOS) {
								window.location.href = 'tel:'+phone_numbers;
								call();
							}else{
								$(".call").show();
								$("#tel").html("请在拨通"+qian+"后<br/>再输入"+hou);
								$("#call").click(function(){
									window.location.href = 'tel:'+phone_numbers;
									call();
								})
							};
						};
					}else{
						window.location.href ='http://'+window.location.host+'/Home/House/login.html?callpage=1&room_id='+room_id+'&url='+url;
					};
				});
				// 底部按钮显示
				var appointbar =  data.show_appointbar;//显示预约
				var telbar = data.show_telbar;//显示电话
				var footerhtml = "";
				if (is_owner == '5' || data.is_regroup == '1') {
					footerhtml += '<div class="zhixun"><a id="zhixun" href="javascript:;">电话咨询</a></div>';
					$("#detailFooter").append(footerhtml);
				}else{
					if (appointbar == 1 && telbar != 1) {
						footerhtml += '<div class="contract"><a class="yuyue" href="javascript:;"  id="js_make">预约</a></div>';
						$("#detailFooter").append(footerhtml);
					}else if(appointbar == 1 && telbar == 1) {
						footerhtml += '<div class="footer_main"><a href="javascript:;" class="appoint yuyue"  id="js_make"><img src="../../Public/images/yu.png"><span>预约</span></a><a href="javascript:;" class="tel lianxi" id="js_abtest">联系房东</p>';
						$("#detailFooter").append(footerhtml);
					}else if(telbar == 1 && appointbar != 1){
						footerhtml += '<div class="contract"><a class="lianxi" href="javascript:;" id="js_abtest">联系房东</a></div>';
						$("#detailFooter").append(footerhtml);
					};
				}
				$(".proDiv").hide();
			};
		}
	});
}
getDate();
// 点击电话咨询
$(document).on("click","#zhixun",function(event){
	$(".Mask").show();
	$(this).css("color","#fff");
	$(document).scrollTop($(".offer").offset().top-($(window).height()-$(".offer").height())/2);
	$(".offer").css("zIndex","23333");
	setTimeout(function(){
		$(".Mask").hide();
		$(".offer").css("zIndex","2330");
	},1000)
});
$(".Mask").click(function(){
	$(this).hide();
	$(".offer").css("zIndex","2330");
})
// 附近房源
$.ajax({
	type:"GET",
	async:false,
	dataType:'json',
	url:http+"housenearby.html",
	data:{room_id:room_id,longitude:room_longitude,latitude:room_latitude,pageno:1,limit:10,sort:-1,city_code:city_code},
	success:function(data){
		console.log(data)
		if (data.status == 200) {
			var data = data.data;//接口数据
			var nearbyhtml = "";
			if (data.length != 0) {
				nearbyhtml += '<div class="nearby_main cf">';
				nearbyhtml += '<div id="nearbySwiper" class="swiper-container swiper-container-horizontal swiper-container-free-mode"><div class="swiper-wrapper">';
				for (var i = 0; i < data.length; i++) {
					nearbyhtml += '<div class="swiper-slide"><a href="http://'+window.location.host+'/'+city_name+'/roomDetail.html?room_id='+data[i].room_id+'&city_code='+city_code+'&city_name='+city_name+'"><img src='+data[i].main_img_path+'><p><span>￥'+data[i].room_money+'</span>&nbsp;元/月</p></a></div>';
				};
				nearbyhtml += '</div></div></div>';
				$("#nearby").append(nearbyhtml);//附近房源数据
			}else{
				$("#nearby").hide();
			};
		};
	}
});
// 点击取消
$("#cancel").click(function(){
	$(".call").hide();
})
//电话统计
var call = function(){
	var date = new Date();
	var time = Date.UTC(date.getFullYear(),date.getMonth(),date.getDay(),date.getHours(),date.getMinutes(),date.getSeconds())/1000;//事件戳
	var urls = {mine_number:user_mobile,virtual_flag:virtual_flag,room_id:room_id,room_no:room_no,ld_number:ld_number,time:time,owner_id:owner_id,owner_name:owner_name,source:0,city_code:city_code};
	$.ajax({
		type:"GET",
		async:false,
		dataType:'json',
		url:http+"rentercall.html",
		data:urls,
		success:function(data){
			if (data.status == '200') {
			};
		}
	})
};
// 点击拨打电话
$(document).on("click",".lianxi",function(){
	var status = getCookie("apicustomer_id") || '';//判断用户是否登录
	var qian = ld_number.substr(0,10);
	var hou = ld_number.substr(11);
	if (status.length > 0) {
		if (userNumber == "1") {
			$.ajax({
				type:"GET",
				async:false,
				dataType:'json',
				url:http+"virtualcallno.html",
				data:{phone_number:user_mobile,virtual_number_id:number_id,room_id:room_id,room_no:room_no,city_code:city_code},
				success:function(data){

					var phone = data.data.virtual_numbers;//得到虚拟号
						virtual_flag = data.data.virtual_flag;//虚拟号码标记
						ld_number = phone;
					if(isiOS){
						window.location.href = 'tel:'+phone;
							call();
					}else{
						if (phone.substr(0,3) == '400') {
							var phone_qian = phone.substr(0,10);
							var phone_hou = phone.substr(11);
							$(".call").show();
							$("#tel").html("请在拨通"+phone_qian+"后<br/>再输入"+phone_hou);
							$("#call").click(function(){
								window.location.href = 'tel:'+phone;
									call();
							})
						}else{
							window.location.href = 'tel:'+phone;
								call();
						};
					}
					// $(".call").show();
					// $("#tel").html(phone);
					// $("#call").attr("href","tel:"+phone);
				}
			})
		}else{
			virtual_flag = -1;
			if (isiOS) {
				$(".lianxi").attr("href",'tel:'+ld_number);
				$(".lianxi").attr("phone",ld_number);
				window.location.href = 'tel:'+ld_number;
				call();
			}else{
				$(".call").show();
				$("#tel").html("请在拨通"+qian+"后<br/>再输入"+hou);
				$("#call").click(function(){
					$(".lianxi").attr("href",'tel:'+ld_number);
					$(".lianxi").attr("phone",ld_number);
					window.location.href = 'tel:'+ld_number;
					call();
				})
			};
		}
	}else{
		window.location.href ='http://'+window.location.host+'/Home/House/login.html?callpage=1&room_id='+room_id+'&url='+url;
	};
})
// 点击预约
$(document).on("click",".yuyue",function(){
	var status = getCookie("apicustomer_id") || '';//判断用户是否登录
	if (status.length > 0) {
		if (has_appointed == "1") {
			$(".common_box").show().find("p").text(alert_text_cannot);
			setTimeout(function(){
				$(".common_box").fadeOut();
			},500);
		}else{
			window.location.href ='http://'+window.location.host+'/Home/House/appoint.html?room_id='+room_id+'&url='+url;
		};
	}else{
		window.location.href ='http://'+window.location.host+'/Home/House/login.html?appoint=1&room_id='+room_id+'&url='+url;
	};
})