
// Solution to load an external JS script
function importScript(scriptPath="") {
	var imported = document.createElement('script');
	imported.src = scriptPath;
	document.head.appendChild(imported);	
};

//Solution to load an external CSS stylesheet
function importCSS(scriptPath="") {
	var ss = document.createElement('link');
	ss.type = "text/css";
	ss.rel = "stylesheet";
	ss.href = scriptPath;
	document.getElementsByTagName("head")[0].appendChild(ss);	
};

//Testar se um objeto existe
function isEmptyObject(obj) {
	var name;
	for (name in obj) {
		return false;
	}
	return true;
};

function makeid() {
	var text = "";
	var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

	for (var i = 0; i < 5; i++) {
		text += possible.charAt(Math.floor(Math.random() * possible.length));
	};

	return text;
}

function addDiv(divname, destination, linechart1) {
	//Add the div and tabs
	var newDivName = divname+makeid();
	var newfsName = "fs_"+divname+makeid();
	
	var data = linechart1.data();			
	var meanValue = linechart1.mean();
	var currentFeature = $('#featuresPlayer').val();
	var currentDate = $('#daylist').val();
	
	var aspas = '"';
	
	$('#'+destination).append(
			"<fieldset id='"+newfsName+"' style='text-align:left;'><legend><b>"+currentFeature+" - "+currentDate+
			"</b></legend><table><tr><td width='100%'><div id='"+newDivName+
			"' style='width:100%; height:130px; float:left;'></div></td><td width='100px;'>" +
			"<img src='./static/images/remove.png' onclick='removeDiv("+aspas+newfsName+aspas+");'>" +
			"</td></tr></table></fieldset>");

	var line2 = linechart()
	    .$el(d3.select("#"+newDivName))
	    .width(linechart1.width()+150)
	    .height(100)
	    .color("black")
	    .transitionTime(30)	    
	    .render();

	line2
		.mean(meanValue)
    	.data(data)
    	.render();
}

function removeDiv(divname) {		
	d3.select('#'+divname).remove();	
}
