
$(document).ready(function () {
  // alert('stop')
  $('#predict_form').on('submit', function (event) {
    event.preventDefault();
    var formData = new FormData(this);
    // alert("toto")

    $.ajax({
      type: 'POST',
      url: '/predict',
      data: formData,
      processData: false,
      contentType: false,
      cache: false,
      success: function (response) {
        var plotData = JSON.parse(response.consomation);
        Plotly.newPlot('graph', plotData.data, plotData.layout);
        console.log(plotData);
      }
    });
  });
});