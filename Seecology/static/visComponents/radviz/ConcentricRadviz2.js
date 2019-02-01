/*########################################################################################################################################
* ### Concentric Radviz - Clausius Duque Reis (clausiusreis@gmail.com #################################################################### 
* ########################################################################################################################################*/

function ConcentricRadviz(options){
	// Parameters
	var container   	= "#"+options.container,
		layers 		= options.layers,
		dataOriginal	= options.data,
         serverPath	= options.serverPath,
         labels       = options.labels,
         samplesOpacity = options.samplesOpacity;
	
    // Intern variables
	var margin = {top: 20, right: 20, bottom: 20, left: 20};
	var svg, width, height, r, radvizData;
	var drag, g, face, rings, sliderGroup, samples;	
	var color = ['#990000','#034e7b','#005a32','#d7301f','#0570b0','#238b45'];

    var colorLabels = ["#aaaaaa", "#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", 
                       "#0099c6", "#dd4477", "#66aa00", "#b82e2e", "#316395","#994499", 
                       "#22aa99", "#aaaa11", "#6633cc", "#e67300", "#8b0707", "#651067", 
                       "#329262", "#5574a6", "#3b3eac"];
    //var colorLabels = ['#adadad','#ea0909','#008411','#0225d3','#3d54c6','#fc8302',
    //                   '#70ff00','#016368','#fff200','#7a7400','#5b0077'];

	//Variáveis globais
	degree1=0;
	degree2=0;
	anchorLayer=-1;
	selectedFeature="";
	
	var rotationLayers = [];
    var radvizLayers = [];
    var radvizLayersSliders = [];
    var numElemByLayer =[];
    var dataSamples = [];    
	
    var object = {};

    function addNormalizedValues(data) {
        data.forEach(function(d) {
            layers.forEach(function(dd) {
                d[dd.name] = +d[dd.name];                
            });
        });
        var normalizationScales = {};
        
        //numlayers = layers[layers.length-1].layer+1;
        auxCorrectionValue = 1.3; // 1.4 Value to keep samples inside the central circle
        
        layers.forEach(function(dd) {
            normalizationScales[dd.name] = d3.scaleLinear().domain(d3.extent(data.map(function(d) {
            	return d[dd.name]*auxCorrectionValue;
            }))).range([ 0, 1 ]);
        });        
        data.forEach(function(d) {
        	layers.forEach(function(dd) {
                d[dd.name] = normalizationScales[dd.name](d[dd.name]);
            });
        });
        return data;
    };

	/*########################################################################################################################################
	* ### Method for render/refresh graph #################################################################################################### 
	* ########################################################################################################################################*/
	object.render = function(){

		if(!svg){ //### RENDER FIRST TIME ###

			data = addNormalizedValues(dataOriginal);
			//console.log("data", data);

			width = d3.select(container).node().clientWidth - margin.left - margin.right;
			height = d3.select(container).node().clientHeight - margin.top - margin.bottom;              
			r = Math.min(width, height)/2; // radius of entire figure
			
			//Generating arc data
			var dataArcCount = [];
			for (i = 0; i < layers.length; i++) {
				if (dataArcCount[layers[i].layer] == null) {
					dataArcCount[layers[i].layer] = 1;
				} else {
					dataArcCount[layers[i].layer] = dataArcCount[layers[i].layer]+=1;
				}
			};
			
			fillLayers(layers);
						
			radvizData = { 
		  	        r: r - (numElemByLayer.length*30),
		  	        faceColor: '#f2f7ff',
		  	        tickColor: '#B2B2B2',
		  	        sliderR: 15      
		  	      };

			fillDataSamples(data, radvizLayers, radvizData.r, radvizLayersSliders);
			
			// drag behavior
			drag = d3.drag()
				.on('start', function(d){
					anchorLayer = d['layer'];
					selectedFeature = d['value'];

					//dragstart;					
					d3.select(this).select('.slider-background')
						.transition()
						.attr('r', radvizData.sliderR);
									
					var deg = ((Math.atan2(d3.event.y, d3.event.x) / Math.PI) * 180);
				    if (deg < 0) deg += 360;

					degree1 = deg;
					degree2 = deg;
				})
				.on('drag', drag);

			svg = d3.select(container).append("svg")
			    .attr("width", width + margin.left + margin.right)
			    .attr("height", height + margin.top + margin.bottom);
			
			g = svg.append('g')
				.attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
		    
			// Circulo central
		    face = g.append('g')
		    	.datum(radvizData)
		      	.attr('transform', 'translate(' + width/2 + ',' + r + ')');
		    
		    // Desenho o círculo central		    
		    face.append('circle')
		      	.attr("r", function(d) { return d.r; })
		      	.style("fill", function(d) { return d.faceColor; })
		      	.style("stroke", function(d) { return d.tickColor; })
		      	.style('stroke-width', 2);

		    // Circulo dos Layers
		    rings = face.selectAll('.outer-ring')
		      	.data(dataArcCount)
		      .enter().append('g')
		      	.classed('outer-ring', true);

		    // Desenho os círculos dos layers
		    rings.append('circle')
		      	.classed('ring', true)
		      	.attr("r", function(d,i) { 
	      			return r-(i*30); 
		      	})
		      	.style("fill", "none")
		        .style("stroke", function(d,i){ return color[i]; })//radvizData.tickColor)
		        .style("stroke-width", radvizData.sliderR/3)
		        .attr("opacity", 0.7);
		    
		    // Sliders e âncoras
		    sliderGroup = face.selectAll('.slider')
		      	.data(radvizLayers)
		      .enter()
		    	.append('g')
		      	.style('cursor', 'move')
		      	.on('mouseover', function(d) {})
		      	.on('mouseleave', function(d) {})
		      	.call(drag);

		    
		    //console.log(radvizLayers);
		    // slider
		    sliderGroup.append('circle')
		    	.classed('slider', true)
		    	.attr("r", radvizData.sliderR)
		    	.style("fill", function(d) { return d.color; })
		    	.style("stroke", "white")
		    	.style("stroke-width", 2)
		    	.attr("cx", function(d) { return d.cx; })
		    	.attr("cy", function(d) { return d.cy; })
		    	.on('mouseover', function(d,i) {
		      		div.transition()        
	                    .duration(200)      
	                    .style("opacity", 1);

	                var string = "<b>"+d.name+"</b>";

	                //Testar a posição do mouse em relação ao RADVIZ e colocar o tooltip acima ou abaixo
	                div .html(string) //this will add the image on mouseover
	                    .style("left", (d3.event.x-50) + "px")     
	                    .style("top", (d3.event.y-40) + "px")
	                    .style("width", "100px")
	                    .style("height", "20px")
	                    .style("font-color", "black");
		      	})
		      	.on('mouseleave', function(d) {
		      		div.transition()        
	                    .duration(200)      
	                    .style("opacity", 0);
		      	});

		    //Ancoras de cada feature
		    sliderGroup.append('circle')
		    	.classed('anchor', true)
				.attr("id", function (d){ return d.value })
				.attr("r", 5)
				.style("fill", function(d) { return d.color; })
		    	.attr("cx", function(d) { return d.cxa; })
		    	.attr("cy", function(d) { return d.cya; });
		      
		    // slider content
		    sliderGroup.append('text')
		    	.classed('content', true)
		      	.text(function(d) { return d.value+1; })
		      	.style("fill", "white")
		      	.style("text-anchor", "middle")
		      	.style("font-size", 16)
		      	.style("font-family", "sans-serif")
		    	.attr("x", function(d) { return d.cx; })
		    	.attr("y", function(d) { return d.cy + 5.5; })
		    	.on('mouseover', function(d,i) {
		      		div.transition()        
	                    .duration(200)      
	                    .style("opacity", 1);

	                var string = "<b>"+d.name+"</b>";

	                //Testar a posição do mouse em relação ao RADVIZ e colocar o tooltip acima ou abaixo
	                div .html(string) //this will add the image on mouseover
	                    .style("left", (d3.event.x-50) + "px")     
	                    .style("top", (d3.event.y-40) + "px")
	                    .style("width", "100px")
	                    .style("height", "20px")
	                    .style("font-color", "black");
		      	})
		      	.on('mouseleave', function(d) {
		      		div.transition()        
	                    .duration(200)      
	                    .style("opacity", 0);
		      	});
			
		    var div = d3.select("body").append("div")	
			    .attr("class", "tooltip")				
			    .style("opacity", 0);
		    		    
		    //DATA Samples
		    samples = face.selectAll('.sample')
		      	.data(dataSamples)
		      .enter()
		    	.append('g')
		      	.style('cursor', 'pointer')
		      	.on('mouseover', function(d,i) {
		      		div.transition()        
	                    .duration(200)      
	                    .style("opacity", 1);

	                var string = "" +
	                		"<img width='200' height='90' src='" + serverPath + "/Extraction/AudioSpectrogram/" + dataOriginal[i].FileName + ".png' /><br>" +
            				"<b>Info do sample.</b><br>" +
            				"Info, info, info...";

	                //Testar a posição do mouse em relação ao RADVIZ e colocar o tooltip a direita ou esquerda
	                div .html(string) //this will add the image on mouseover
	                    .style("left", (d3.event.x-230) + "px")     
	                    .style("top", (d3.event.y-100) + "px")
	                    .style("width", "208px")
	                    .style("height", "120px")
	                    .style("font-color", "white");
		      	})
		      	.on('mouseleave', function(d) {
		      		div.transition()        
	                    .duration(200)      
	                    .style("opacity", 0);
		      	})
		      	.on('click', function(d,i){
		      		//console.log(dataOriginal[i].FileName, dataOriginal[i].SecondsFromStart);
		      		$( "#tabs" ).tabs({ active: 1 });
		      		currentFile = dataOriginal[i].FileName;
		      		updatePlayer1(currentFile, dataOriginal[i].SecondsFromStart);
		      	});
            
		    samples.append('circle')
        		    	.classed('sample', true)
        			.attr("ID", function (d,i){ return dataOriginal[i].FileName+":"+dataOriginal[i].SecondsFromStart; }) // FileName + SecondsFromStart
        			.attr("r", 5)
        		    	.attr("cx", function(d) { return d.cx; })
        		    	.attr("cy", function(d) { return d.cy; })
                  .style("opacity", samplesOpacity)
        		    	.style("fill", function(d,i){    
                      var cor = "";
                      if (labels.length > 0) {
                          cor = colorLabels[labels[i].labelIndex];
                      } else {  
                          cor = colorLabels[0];
                      }
        		    		return cor;
        		    	});
//console.log(dataSamples);
		    //##################################################################################################################################
		    //##################################################################################################################################
		    //##################################################################################################################################
			// Lasso functions
			var lasso_start = function() {
			    lasso.items()
			        //.attr("r",3); // reset size
			        //.classed("not_possible",true)
			        //.classed("selected",false);
			};
			
			var lasso_draw = function() {
			
			    // Style the possible dots
			    lasso.possibleItems().select("circle")
			        //.classed("not_possible",false)
			        //.classed("possible",true)
                         //.attr("r",3)
                         .style("opacity", samplesOpacity)
                         .style("fill", function(d,i){                            
                            return "green";
                         });
                 

			    // Style the not possible dot
			    lasso.notPossibleItems().select("circle")
			        //.classed("not_possible",true)
			        //.classed("possible",false)
                         //.attr("r",3)
                         .style("opacity", samplesOpacity);
                         //.style("fill", function(d,i){
                         //   var cor = "";
                         //   if (labels.length > 0) {
                         //       cor = colorLabels[labels[i].labelIndex];
                         //   } else {  
                         //       cor = colorLabels[0];
                         //   }
                         //   return cor;
                         //});
			};
			
			var lasso_end = function() {
			    // Reset the color of all dots
			    lasso.items().select("circle")
			        //.classed("not_possible",false)
			        //.classed("possible",false)
                         //.attr("r",3)
                         .style("opacity", samplesOpacity)
                         .style("fill", function(d,i){
                            var cor = "";
                            if (labels.length > 0) {
                                cor = colorLabels[labels[i].labelIndex];
                            } else {  
                                cor = colorLabels[0];
                            }
                            return cor;
                         });
			
			    // Style the selected dots
			    dots = lasso.selectedItems().select("circle")
			        //.classed("selected",true)
			        //.attr("r",20)
                         .attr("r",8)
                         .style("opacity", 1)
                         .style("fill", function(d,i){
                            return "red";
                         });
			
                //console.log(dots);

			    // Reset the style of the not selected dots
			    lasso.notSelectedItems()
                     .select("circle")
                     .style("opacity", samplesOpacity)
			        .attr("r",5);
						    			    
			    selElemIDs = [];
			    for (element in dots._groups[0]){			    	
			        //sample = dots._groups[0][element].getElementsByClassName("sample")[0];
                     sample = dots._groups[0][element]; //Get the circle
			        //console.log(sample.getAttribute("ID"));
                     //console.log(sample);
			        selElemIDs.push(sample.getAttribute("ID"));
			    }
			    //console.log( selElemIDs );
			    updateGrid(selElemIDs);
			    $( "#tabs" ).tabs({ active: 1 });
			};
			
			var lasso = d3.lasso()
	            .closePathSelect(true)
	            .closePathDistance(200)
	            .items(samples)
	            .targetArea(svg)
	            .on("start",lasso_start)
	            .on("draw",lasso_draw)
	            .on("end",lasso_end);
			
			svg.call(lasso);
		    //##################################################################################################################################
		    //##################################################################################################################################
		    //##################################################################################################################################
		    
	    }else{ //### REFRESH DATA ###
	    	//object.data(data);
	    }

		return object;
	};

	/*########################################################################################################################################
	* ### Visualization Functions ############################################################################################################ 
	* ########################################################################################################################################*/
	
	function fillLayers(layers){
		for (i=0; i<layers.length; i++){
			radvizLayers.push({"value":layers[i].value, "name":layers[i].name, "layer":layers[i].layer, "degree":0, "cx":0, "cy":0, "cxa":0, "cya":0, "color":"white"});
		}

		//Conto quantos elementos tenho em cada layer
		rotationLayers = [];
		numElemByLayer = [];
	    for (i=0; i<radvizLayers.length; i++){		
	    	if (numElemByLayer[radvizLayers[i].layer] == null){
	    		numElemByLayer[radvizLayers[i].layer] = 0;
	    		rotationLayers[radvizLayers[i].layer] = 0;
	    	}
	    	numElemByLayer[radvizLayers[i].layer] += 1;    	
	    }
	    
	    //Preencho o ângulo inicial e cor de cada feature
	    var currentLayer = -1;
	    var elemID = 0;
	    for (i=0; i<radvizLayers.length; i++){		
			if (currentLayer != radvizLayers[i].layer) {
				currentLayer = radvizLayers[i].layer;
				elemID = 0;
			}
	    	var gap = 360 / numElemByLayer[currentLayer];
	    	var cr = r-(currentLayer*30)
	    	
	    	radvizLayers[i].degree = gap * elemID;
	    	radvizLayers[i].color = color[currentLayer];

	    	radvizLayers[i].cx = (r-(currentLayer*30)) * Math.cos(radvizLayers[i].degree * (Math.PI / 180));
	    	radvizLayers[i].cy = (r-(currentLayer*30)) * Math.sin(radvizLayers[i].degree * (Math.PI / 180));
	    	radvizLayers[i].cxa = (r-(numElemByLayer.length*30)) * Math.cos(radvizLayers[i].degree * (Math.PI / 180));
	    	radvizLayers[i].cya = (r-(numElemByLayer.length*30)) * Math.sin(radvizLayers[i].degree * (Math.PI / 180));

	    	//Sliders
	    	radvizLayersSliders[i] = 1;
	    	
	    	elemID += 1;
	    }
	}

	function fillDataSamples(data, radvizLayers, r, radvizLayersSliders){
		
		//console.log('Clausius', radvizLayers[radvizLayers.length-1].layer+1);
		//numLayers = radvizLayers[radvizLayers.length-1].layer+1;
		
		for (i=0; i<data.length; i++){
			auxVec = [];
			cx = 0;
			cy = 0;
			len = 0;
			for (j=0; j<radvizLayers.length; j++){
				name = radvizLayers[j].name;
				value = data[i][radvizLayers[j].name];
				auxVec[name] = value;
				cx += (radvizLayers[j].cxa * value * radvizLayersSliders[j]);
				cy += (radvizLayers[j].cya * value * radvizLayersSliders[j]);
				len += radvizLayers[j].cxa*radvizLayers[j].cxa + radvizLayers[j].cya*radvizLayers[j].cya;
			}
			len = Math.sqrt(len);			
			auxVec["cx"] = (cx/len)*((r)*1.6);
			auxVec["cy"] = (cy/len)*((r)*1.6);
	
			dataSamples.push(auxVec);
		}
	}
	
	function updateDataSamples(dataSamplesIn, radvizLayers, r, radvizLayersSliders){
	
		//numLayers = radvizLayers[radvizLayers.length-1].layer+1;
		
		for (i=0; i<dataSamplesIn.length; i++){
			cx = 0;
			cy = 0;
			len = 0;			
			for (j=0; j<radvizLayers.length; j++){
				value = dataSamplesIn[i][radvizLayers[j].name];				
				cx += (radvizLayers[j].cxa * value * radvizLayersSliders[j]);
				cy += (radvizLayers[j].cya * value * radvizLayersSliders[j]);
				len += radvizLayers[j].cxa*radvizLayers[j].cxa + radvizLayers[j].cya*radvizLayers[j].cya;
			}
			len = Math.sqrt(len);			
			dataSamples[i].cx = (cx/len)*((r)*1.6);
			dataSamples[i].cy = (cy/len)*((r)*1.6);
		}
	}
	
	function drag() {
		
		var deg = ((Math.atan2(d3.event.y, d3.event.x) / Math.PI) * 180);
	    if (deg < 0) deg += 360;

        //Calculo a rotação em degrees
        degree2 = deg;
        var rotation = degree2 - degree1;
        degree1 = deg;
        degree2 = deg;
        rotationLayers[anchorLayer] += rotation;

        // Converts from degrees to radians.
        Math.radians = function(degrees) {
          return degrees * Math.PI / 180;
        };
         
        // Converts from radians to degrees.
        Math.degrees = function(radians) {
          return radians * 180 / Math.PI;
        };

	    for (i=0; i<radvizLayers.length; i++){
	    	if (radvizLayers[i].layer == anchorLayer){

	    		if (d3.event.sourceEvent.shiftKey) {
	    			if (radvizLayers[i].value == selectedFeature) {
	    				radvizLayers[i].degree += rotation;
			    		radvizLayers[i].cx = (r-(radvizLayers[i].layer*30)) * Math.cos(Math.radians(radvizLayers[i].degree));
				    	radvizLayers[i].cy = (r-(radvizLayers[i].layer*30)) * Math.sin(Math.radians(radvizLayers[i].degree));
				    	radvizLayers[i].cxa = (r-(numElemByLayer.length*30)) * Math.cos(Math.radians(radvizLayers[i].degree));
				    	radvizLayers[i].cya = (r-(numElemByLayer.length*30)) * Math.sin(Math.radians(radvizLayers[i].degree));
	    			}
	    		} else {
		    		radvizLayers[i].degree += rotation;
		    		radvizLayers[i].cx = (r-(radvizLayers[i].layer*30)) * Math.cos(Math.radians(radvizLayers[i].degree));
			    	radvizLayers[i].cy = (r-(radvizLayers[i].layer*30)) * Math.sin(Math.radians(radvizLayers[i].degree));
			    	radvizLayers[i].cxa = (r-(numElemByLayer.length*30)) * Math.cos(Math.radians(radvizLayers[i].degree));
			    	radvizLayers[i].cya = (r-(numElemByLayer.length*30)) * Math.sin(Math.radians(radvizLayers[i].degree));
	    		}
		    	
	    	}
	    }

	    //console.log("AfterDrag radvizLayers", radvizLayers);
	    
        //Arco do feature
	    d3.selectAll('.slider')
        	.attr("cx", function(d,i) { return radvizLayers[i].cx; })
        	.attr("cy", function(d,i) { return radvizLayers[i].cy; });
        
      	//Ancora do feature
        d3.selectAll('.anchor')
        	.attr("cx", function(d,i) { return radvizLayers[i].cxa; })
        	.attr("cy", function(d,i) { return radvizLayers[i].cya; });

        //Nome do feature 
        d3.selectAll('.content')
        	.attr("x", function(d,i) { return radvizLayers[i].cx; })
        	.attr("y", function(d,i) { return radvizLayers[i].cy + 5.5; });
	    
        
        updateDataSamples(dataSamples, radvizLayers, radvizData.r, radvizLayersSliders);        
        
        var update = d3.selectAll(".sample")
        	.data(dataSamples)
        	.attr("cx", function(d) { return d.cx; })
	    	.attr("cy", function(d) { return d.cy; });
        	
        
	}
	
	/*########################################################################################################################################
	* ### Getters/Setters #################################################################################################################### 
	* ########################################################################################################################################*/
	
	return object;
};
