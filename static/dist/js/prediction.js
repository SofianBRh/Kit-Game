
$(document).ready(function () {
    // alert('stop')
    const loader = document.getElementById('loader');

    const fileInput = document.getElementById("file");
    const filenameLabel = document.getElementById("filename");
    fileInput.addEventListener("change", () => {
        const filename = fileInput.files[0].name;
        filenameLabel.textContent = filename;
    });

    $('#predict_form').on('submit', function (event) {
        event.preventDefault();
        var formData = new FormData(this);
        // alert("toto")
        loader.classList.remove('hidden');
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: formData,
            processData: false,
            contentType: false,
            cache: false,
            success: function (response) {
                loader.classList.add('hidden');
                var plotData = JSON.parse(response.consomation);
                var plotData_temperature = JSON.parse(response.temperature);
                Plotly.newPlot('graph', plotData.data, plotData.layout);
                Plotly.newPlot('graph_temperature', plotData_temperature.data, plotData_temperature.layout);
                console.log(plotData);
            }
        });
    });
});