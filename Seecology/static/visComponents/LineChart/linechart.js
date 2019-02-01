/*#############################################################################################################################################
* ### Define line graph ####################################################################################################################### 
* #############################################################################################################################################*/
function linechart(){
	// Default settings
	var $el = d3.select("body")
	var width = 960;
	var height = 500;
	var color = "steelblue";
	var margin = {top: 10, right: 10, bottom: 30, left: 50};
	var data = [];	
	var transitionTime = 200;
	var currentSampleStart = -1;
	var currentSampleEnd = -1;
	var mean = -1;
	
	var svg, x, y, yy, yyData, xAxis, yAxis, line, line1, lineMean, ttdiv, rect;
	var object = {};        

	var interpolatorOptions = ["StepAfter", "MonotoneX"];

	//var formatDate = d3.timeFormat("%Y-%m-%d %H:%M:%S");
	var formatTime = d3.timeFormat("%H:%M");
	var parseDate = d3.timeParse("%Y-%m-%d %H:%M:%S");
	
	/*########################################################################################################################################
	* ### Method for render/refresh graph #################################################################################################### 
	* ########################################################################################################################################*/
	object.render = function(){
		if(!svg){ //### RENDER FIRST TIME ###
			
			//Definition of scales 
			//x = d3.scaleLinear().range([0, width]);
			x = d3.scaleLinear().range([0, width]);
			y = d3.scaleLinear().range([height, 0]);
			yy = d3.scaleLinear()
				.range(d3.extent(data, function(d) { return d.y; }))
				.domain([height, 0]);			
			yyData = d3.scaleLinear()
				.range(d3.extent(data, function(d) { return d.x; }))
				.domain([0, width]);
			
			//Definition of the axis object
			xAxis = d3.axisBottom().scale(x);
			yAxis = d3.axisLeft().scale(y).ticks(6);

			//Definition of a line object
			line = d3.line()
				.x(function(d) { return x(d.x); })
				.y(function(d) { return y(d.y); })
                  .curve(d3.curveStepAfter);
				//.curve(d3.curveMonotoneX);

            //Mean line
			if (mean == -1) {
	            meanValue = d3.mean(data, function(d) { return +d.y });
	            mean = meanValue;
	            lineMean = d3.line()
					.x(function(d) { return x(d.x); })
					.y(function(d) { return y(meanValue); });
			} else {            
	            lineMean = d3.line()
					.x(function(d) { return x(d.x); })
					.y(function(d) { return mean; });
			}

			//Append the SVG on the target DIV
			svg = $el.append("svg")
				.attr("width", width + margin.left + margin.right)
				.attr("height", height + margin.top + margin.bottom)
				.append("g")
				.attr("transform", "translate(" + margin.left + "," + margin.top + ")")
				.style("cursor", "pointer");
//				.on("click", function(d,i){
//					var mouse = d3.mouse(this);
//					mean = yy(mouse[1]);
//					object.mean(yy(mouse[1])).render();
//				});		
			
//			// Define the div for the tooltip
            ttdiv = $el.append("div")
                    .attr("class", "tooltip")
                    .style("opacity", 0);
			
			var rect = svg.append("rect")
				.attr("x", 0)
				.attr("y", 0)
				.attr("width", width)
				.attr("height", height)
				.attr("fill", "white")
				//.attr("opacity", 0.01)
	            //.on("mouseon", showTooltip)
	            .on("mouseout", hideTooltip)
	            .on("mousemove", mouseHover)
	            .on("click", function(d,i){
	            	var curFile = data[Math.floor(yyData(d3.mouse(this)[0]))].filename;
	            	var curSample = data[Math.floor(yyData(d3.mouse(this)[0]))].secondsFromStart;
	            	var curFeature = data[Math.floor(yyData(d3.mouse(this)[0]))].feature;
	            	//console.log(data);
	            	
	            	currentFile = curFile;
	            	currentSample = curSample;
	            	
//	            	var formatDate1 = d3.timeFormat("%Y.%m.%d_%H.%M.%S");
//	            	var parseDate1 = d3.timeParse("%Y-%m-%d %H:%M:%S");
	            	
	            	console.log(curFile, curSample, curFeature);
	            	
	            	updatePlayer(curFile, curSample, curFeature, 0);
//	            	
	            	SelectElement("featuresList", currentFeature);
	            	
	            	document.getElementById('light').style.display='block';
	            	document.getElementById('fade').style.display='block';
				});
//				.on("click", function(d,i){
//					var mouse = d3.mouse(this);
//					object.mean(yy(mouse[1])).render();
//				});

			//Draw the Circle
			//var circle1 = svg.append("circle").attr("cx", -100).attr("cy", -100).attr("r", 20);			
			var circle = svg.append("circle").attr("cx", -100).attr("cy", -100).attr("r", 20);
			//var circle1 = svg.append("circle").attr("cx", -100).attr("cy", -100).attr("r", 20);
			//var square = svg.append("rect").attr("x", -100).attr("y", -100).attr("width", 40).attr("height", 20).attr("fill", "red");
			var line1 = svg.append("line").attr("x1", 0).attr("y1", 0).attr("x2", 1).attr("y2", 1);
			var line2 = svg.append("line").attr("x1", 0).attr("y1", 0).attr("x2", 1).attr("y2", 1);
            function mouseHover() {            	
            	//console.log(d3.mouse(this)[0], d3.mouse(this)[1]);            	
            	
            	//console.log(data[]);
            	
//            	square.remove();
            	circle.remove();
            	line1.remove();
            	line2.remove();

//            	circle1 = svg.append("circle")
//            		.data(data)
//					.attr("cx", d3.mouse(this)[0])
//					.attr("cy", function(d,i) { return y(1); })
//					.attr("r", 10)
//					.attr("fill", "red");
            	
//            	square = svg.append("rect")
//            		.attr("x", d3.mouse(this)[0]-39)
//            		.attr("y", function(d,i) { return y(1)-10; })
//            		.attr("width", 40)
//            		.attr("height", 20)
//            		.attr("fill", "red");
//            	            	
            	circle = svg.append("circle")
					.attr("cx", d3.mouse(this)[0])
					.attr("cy", function(d,i) { return y( data[Math.floor(yyData(d3.mouse(this)[0]))].y ); })
					.attr("r", 6)
					.attr("fill", "red")
					.style("pointer-events", "none");
            	
            	line1 = svg.append("line")
	            	.attr("x1", d3.mouse(this)[0])
	            	.attr("y1", function(d,i) { return y(0); })
	            	.attr("x2", d3.mouse(this)[0])
	            	.attr("y2", function(d,i) { return y(1.01); })
	            	.attr("stroke", "gray")
	            	.attr("opacity", 0.4)
	            	.attr("stroke-width", 2)
	            	.style("stroke-dasharray", "3")
	            	.style("pointer-events", "none");

            	line2 = svg.append("line")
	            	.attr("x1", x(0))
	            	.attr("y1", d3.mouse(this)[1])	            	
	            	.attr("x2", d3.max(data, function(d) { return d.x; } ))
	            	.attr("y2", d3.mouse(this)[1])
	            	.attr("stroke", "gray")
	            	.attr("opacity", 0.4)
	            	.attr("stroke-width", 2)
	            	.style("stroke-dasharray", "3")
	            	.style("pointer-events", "none");
            	
            	var mydiv = this.getBoundingClientRect();
            	//console.log(mydiv.top, mydiv.right, mydiv.bottom, mydiv.left);
            	
            	ttdiv.style("opacity", 1).style("pointer-events", "none");
            	ttdiv.html(            				
            				"<strong>Recording:</strong> " + data[Math.floor(yyData(d3.mouse(this)[0]))].datetime + "</br>" +
	                        "<strong>" + data[Math.floor(yyData(d3.mouse(this)[0]))].feature + ":</strong> " + (data[Math.floor(yyData(d3.mouse(this)[0]))].y).toFixed(4) + "</br>" +
	                        "<strong>Seconds From Start:</strong> " + data[Math.floor(yyData(d3.mouse(this)[0]))].secondsFromStart + "</br>"
		                )
		                .style("left", (mydiv.left+d3.mouse(this)[0]+20) + "px")
		                .style("top", (mydiv.top+d3.mouse(this)[1]-40) + "px");
            };

//            function showTooltip() {
//            	ttdiv.style("opacity", .85);
//            };
            
            function hideTooltip() {
            	ttdiv.style("opacity", 0);
            };
            
			//Set the extension of the axis values
			x.domain(d3.extent(data, function(d) { return d.x; }));
			//y.domain(d3.extent(data, function(d) { return d.y; }));
            y.domain([0,1]);

            //Draw the lines
			svg.append("path")
				.datum(data)
				.attr("class", "line")				
				.attr("d", line)
	            .style("stroke", color)
	            .style("fill", "none")
	            .style("stroke-width", "1px")
	            .style("pointer-events", "none");
            
//            svg.append("path")
//				.datum(data)
//				.attr("class", "line")				
//				.attr("d", line1)
//	            .style("stroke", "blue")
//	            .style("fill", "none")
//	            .style("stroke-width", "1px");

	        //Add the mean line
            svg.append("path")
				.datum(data)
				.attr("class", "line-mean")				
				.attr("d", lineMean)
                .style("stroke", "red")
                .style("fill", "none")
                .style("stroke-width", "2px")
                .style("stroke-dasharray", "5")
                .style("opacity", ".7")
                .style("pointer-events", "none");
                        
			//Add X axis
			svg.append("g")
				.attr("class", "x axis")
				.attr("transform", "translate(0," + height + ")")
				.call(xAxis);

			//Add Y axis
			svg.append("g")
				.attr("class", "y axis")
				.call(yAxis);
			
			//Add Axis Text
			svg.append("text")
			    .attr("x", width/2-22)
			    .attr("y", height+25)
			    .attr("dy", ".28em")
			    .text("Time");
			
			// Change the axis style (Chubby font problem)
			svg.selectAll('.axis line, .axis path')
				.style("stroke", "gray")
                .style("fill", "none")
                .style("stroke-width", "1.5px")
                .style("shape-rendering", "crispEdges");                

	    }else{ //### REFRESH DATA ###
	    	//Set the new data	    	
	    	object.data(data);
	    	
	    	//console.log("Clausius", data[0].datetime, data[data.length-2].datetime);
	    	
			xScale_sec_time_axis = d3.scaleLinear()
				.domain([parseDate(data[0].datetime), parseDate(data[data.length-2].datetime)])
				.rangeRound([0, width]);
	    	
	    	//Set the extension of the axis values
	    	x.domain(d3.extent(data, function(d) { return d.x; }));
	    	//y.domain(d3.extent(data, function(d) { return d.y; }));
	    	y.domain([0,1]);
	    	yy = d3.scaleLinear()
				//.range(d3.extent(data, function(d) { return d.y; }))
	    		.range([0,1])
				.domain([height, 0]);	    	
	    	
	    	yyData = d3.scaleLinear()
				.range(d3.extent(data, function(d) { return d.x; }))
				.domain([0, width]);			
	    	
	    	//Define the transition time for the X axis
	    	svg.select("g.y")
	        	.transition()
	        	.duration(transitionTime)
	        	.call(yAxis);

	    	//Define the transition time for the Y axis
	    	svg.select("g.x")
	        	.transition()
	        	.duration(transitionTime)
	        	//.call(xAxis);
	        	.call(d3.axisBottom(xScale_sec_time_axis)
        			.ticks(12)
        			.tickFormat(formatTime));
	
	    	//Draw the lines
	    	svg.selectAll("path.line")
	    		.datum(data)
	    		.transition()
	    		.duration(transitionTime)
	    		.attr("d", line);
	    	
//	    	svg.selectAll("path.line")
//	    		.datum(data)
//	    		.transition()
//	    		.duration(transitionTime)
//	    		.attr("d", line1);

            //Mean line
	    	if (mean == -1) {
		    	meanValue = d3.mean(data, function(d) { return +d.y });
		    	object.mean(meanValue);		    	
	            svg.selectAll("path.line-mean")
		    		.datum(data)
		    		.transition()
		    		.duration(transitionTime)
		    		.attr("d", lineMean);
	    	} else {
	    		lineMean = d3.line()
					.x(function(d) { return x(d.x); })
					.y(function(d) { return y(mean); });
	    		svg.selectAll("path.line-mean")
		    		.datum(data)
		    		.transition()
		    		.duration(transitionTime)
		    		.attr("d", lineMean);
	    	}
	    	
//	    	console.log("meanValue:", meanValue);
//	    	console.log("mean:", mean);
	    }

		return object;
	};

	//Getter and setter methods
	object.data = function(value){
		if (!arguments.length) return data;
		data = value;
		return object;
	};

	object.$el = function(value){
		if (!arguments.length) return $el;
		$el = value;
		return object;
	};

	object.width = function(value){
		if (!arguments.length) return width;
		width = value;
		return object;
	};

	object.height = function(value){
		if (!arguments.length) return height;
		height = value;
		return object;
	};

	object.color = function(value){
		if (!arguments.length) return color;
		color = value;
		return object;
	};
	
	object.x = function(value){
		if (!arguments.length) return x;
		x = value;
		return object;
	}

	object.transitionTime = function(value){
		if (!arguments.length) return transitionTime;
		transitionTime = value;
		return object;
	}
	
	object.currentSampleStart = function(value){
		if (!arguments.length) return currentSampleStart;
		currentSampleStart = value;
		return object;
	}
	
	object.currentSampleEnd = function(value){
		if (!arguments.length) return currentSampleEnd;
		currentSampleEnd = value;
		return object;
	}
	
	object.mean = function(value){
		if (!arguments.length) return mean;
		mean = value;
		return object;
	}
	
	return object;
};