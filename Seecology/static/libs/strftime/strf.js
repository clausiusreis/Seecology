var it = $("#this-is-it-baby");

$("#buttons > div > div, #chars > div:not('#clear')").click(function() {

	it.val(it.val() + $(this).data("value"));

});

$("#clear").click(function() {
	it.val("");
});

	var readyStateCheckInterval = setInterval(function() {
	if (document.readyState === "complete") {
		clearInterval(readyStateCheckInterval);

    var updateHeader = function() {
      var h1s = document.getElementsByTagName('h1');
      var format = document.getElementById('this-is-it-baby');

      if (h1s.length > 0) {
        if (format.value.length > 0) {
          h1s[0].innerHTML = strftime(format.value);
        } else {
          h1s[0].innerHTML = "STRFTIME";
        }

      }
      setTimeout(updateHeader, 250);
    };

    setTimeout(updateHeader, 1000);

	}
		}, 10);